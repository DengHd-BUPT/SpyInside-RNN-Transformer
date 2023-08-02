import numpy as np
import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from relaxation_Transformer import bounds as LP_bounds
from relaxation_Transformer import ibp_bounds as IBP_bounds
from relaxation_Transformer import LB_split, UB_split
from transformers import GPT2Model, GPT2Config
from tqdm import tqdm

from utils import get_default_device
dev = get_default_device()
print("the device: " + str(dev))

class DeepPoly:
    def __init__(self, lb, ub, lexpr, uexpr, device=None):
        self.lb = lb
        self.ub = ub
        self.lexpr = lexpr
        self.uexpr = uexpr
        assert not torch.isnan(self.lb).any()
        assert not torch.isnan(self.ub).any()
        assert lexpr is None or (
            (not torch.isnan(self.lexpr[0]).any())
            and (not torch.isnan(self.lexpr[1]).any())
        )
        assert uexpr is None or (
            (not torch.isnan(self.uexpr[0]).any())
            and (not torch.isnan(self.uexpr[1]).any())
        )
        self.dim = lb.size()[0]
        self.device = self.lb.device if device is None else device

    @staticmethod
    def deeppoly_from_perturbation(x, eps, truncate=None):
        assert eps >= 0, "epsilon must not be negative value"
        if truncate is not None:
            lb = x - eps
            ub = x + eps
            lb[lb < truncate[0]] = truncate[0]
            ub[ub > truncate[1]] = truncate[1]
            return DeepPoly(lb, ub, None, None)

        else:
            return DeepPoly(x - eps, x + eps, None, None)

    @staticmethod
    def deeppoly_from_dB_perturbation(x, eps_db):
        dBx = torch.log10(torch.abs(x).max()) * 20.0
        dBd = eps_db + dBx
        delta = 10 ** (dBd.item() / 20.0)
        return DeepPoly.deeppoly_from_perturbation(x, delta)

class DPBackSubstitution:
    def _get_lb(self, expr_w, expr_b):
        if len(self.output_dp.lexpr[0].size()) == 2:
            res_w = (
                positive_only(expr_w).t() @ self.output_dp.lexpr[0]
                + negative_only(expr_w).t() @ self.output_dp.uexpr[0]
            )
        else:
            res_w = (
                positive_only(expr_w) * self.output_dp.lexpr[0]
                + negative_only(expr_w) * self.output_dp.uexpr[0]
            )
        res_b = (
            positive_only(expr_w) @ self.output_dp.lexpr[1]
            + negative_only(expr_w) @ self.output_dp.uexpr[1]
            + expr_b
        )

        if self.prev_layer == None:
            result =  (
                positive_only(res_w) @ self.input_dp.lb
                + negative_only(res_w) @ self.input_dp.ub
                + res_b
            )
            print("1")
            return result
        else:
            return self.prev_layer._get_lb(res_w, res_b)

    def _get_ub(self, expr_w, expr_b):
        if len(self.output_dp.lexpr[0].size()) == 2:
            res_w = (
                positive_only(expr_w) @ self.output_dp.uexpr[0]
                + negative_only(expr_w) @ self.output_dp.lexpr[0]
            )
        else:
            res_w = (
                positive_only(expr_w) * self.output_dp.uexpr[0]
                + negative_only(expr_w) * self.output_dp.lexpr[0]
            )
        res_b = (
            positive_only(expr_w) @ self.output_dp.uexpr[1]
            + negative_only(expr_w) @ self.output_dp.lexpr[1]
            + expr_b
        )

        if self.prev_layer == None:
            return (
                positive_only(res_w) @ self.input_dp.ub
                + negative_only(res_w) @ self.input_dp.lb
                + res_b
            )
        else:
            return self.prev_layer._get_ub(res_w, res_b)

// formal model
class Linear(nn.Linear, DPBackSubstitution):
    def __init__(self, in_features, out_features, bias=True, prev_layer=None):
        super(Linear, self).__init__(in_features, out_features, bias)
        self.prev_layer = prev_layer
        # dp: deepPoly 多面体
        self.input_dp = None
        self.output_dp = None

    @staticmethod
    def convert(layer, prev_layer=None, device=torch.device("cuda:0")):
        l = Linear(
            layer.in_features, layer.out_features, layer.bias is not None, prev_layer
        )
        l.weight.data = layer.weight.data.to(device)
        l.bias.data = layer.bias.data.to(device)
        return l

    def assign(self, weight, bias=None, device=torch.device("cuda:0")):
        assert weight.size() == torch.Size([self.out_features, self.in_features])
        assert bias is None or bias.size() == torch.Size([self.out_features])
        self.weight.data = weight.data.t().to(device)
        if bias is not None:
            self.bias.data = bias.data.to(device)
        else:
            self.bias.data = torch.zeros(self.out_features).to(device)

    def forward(self, prev_dp):
        # Initial layer
        if self.prev_layer == None:
            self.input_dp = prev_dp
            lb = (
                 self.input_dp.lb @ positive_only(self.weight)
                 + self.input_dp.ub @ negative_only(self.weight)
                 + self.bias
            )
            ub = (
                self.input_dp.ub @ positive_only(self.weight)
                + self.input_dp.lb @ negative_only(self.weight)
                + self.bias
            )

        # Intermediate layer 中间层
        # 运用反向置换方法，通过前一层的神经元计算边界
        else:
            lb = self.prev_layer._get_lb(self.weight, self.bias)
            ub = self.prev_layer._get_ub(self.weight, self.bias)

        self.output_dp = DeepPoly(
            lb=lb,
            ub=ub,
            lexpr=(self.weight, self.bias),
            uexpr=(self.weight, self.bias),
            device=prev_dp.device,
        )

        return self.output_dp

class ReLU(nn.ReLU, DPBackSubstitution):
    def __init__(self, inplace=False, prev_layer=None):
        super(ReLU, self).__init__(inplace)
        self.prev_layer = prev_layer
        self.output_dp = None

    def forward(self, prev_dp):
        dim = prev_dp.dim
        dev = prev_dp.device
        lexpr_w = torch.zeros(dim).to(device=dev)
        lexpr_b = torch.zeros(dim).to(device=dev)
        uexpr_w = torch.zeros(dim).to(device=dev)
        uexpr_b = torch.zeros(dim).to(device=dev)

        # intermediate layer
        for i in range(dim):
            l, u = prev_dp.lb[i], prev_dp.ub[i]
            if l > 0:
                lexpr_w[i] = 1
                uexpr_w[i] = 1
            elif u < 0:
                pass
            else:
                lexpr_w[i] = 1 if -l < u else 0
                uexpr_w[i] = u / (u - l)
                uexpr_b[i] = -l * u / (u - l)

        lb = self.prev_layer._get_lb(torch.diag(lexpr_w), lexpr_b)
        ub = self.prev_layer._get_ub(torch.diag(uexpr_w), uexpr_b)
        self.output_dp = DeepPoly(
            lb=lb,
            ub=ub,
            lexpr=(lexpr_w, lexpr_b),
            uexpr=(uexpr_w, uexpr_b),
            device=prev_dp.device,
        )

        return self.output_dp

class Gelu_new(nn.Module, DPBackSubstitution):
    def __init__(self, prev_layer=None, method="lp"):
        super(Gelu_new, self).__init__()
        self.prev_layer = prev_layer
        self.output_dp = None
        if method in ["opt", "lp", "ibp"]:
            self.method = method
            # if method == "opt":
            #     self.set_lambda(device)
        else:
            raise RuntimeError(f"not supported bounding method: {method}")
    def gelu_new(self, x):
        return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x**3.0))))
        # return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


    def forward(self, prev_dp):
        dim = prev_dp.dim
        dev = prev_dp.device
        size1 = int(prev_dp.lb.size()[:-1][-1])
        w_x = torch.ones(dim).to(device=dev)
        b_x = torch.zeros(dim).to(device=dev)
        w_m = torch.ones(dim).to(device=dev)
        b_m = torch.zeros(dim).to(device=dev)

        # y = math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
        self.lx = prev_dp.lb
        self.ux = prev_dp.ub
        self.ly = math.sqrt(2.0 / math.pi) * (self.lx + 0.044715 * torch.pow(self.lx, 3.0))
        self.uy = math.sqrt(2.0 / math.pi) * (self.ux + 0.044715 * torch.pow(self.ux, 3.0))

        size2 = 2048
        # size3 = int(self.lx[0][0].size())
        size = size1 * size2
        self.precal_bnd = [{}, {}, {}]
        coeff = torch.zeros(6, size1, int(size2/340)).to(dev)#
        lb = torch.zeros(22, 2048).to(dev)
        ub = torch.zeros(22, 2048).to(dev)
        lerror = torch.zeros(22, 2048).to(dev)
        uerror = torch.zeros(22, 2048).to(dev)
        j = 0
        start = time.perf_counter()
        for d in range(size1):
            for s in range(50, 2048, 340):#

                #if (self.lx[0][d][s] != self.ux[0][d][s]) and (self.ly[0][d][s] != self.uy[0][d][s]):
                (
                    coeff[0, d, j],
                    coeff[1, d, j],
                    coeff[2, d, j],
                    coeff[3, d, j],
                    coeff[4, d, j],
                    coeff[5, d, j],
                    lb[d][s],
                    ub[d][s],
                    lerror[d][s],
                    uerror[d][s],
                ) = LP_bounds(
                    self.lx[0][d][s],
                    self.ux[0][d][s],
                    self.ly[0][d][s],
                    self.uy[0][d][s],
                    gelu=True,
                )

                j = j+1
            j = 0
        end1 = time.perf_counter()
        print(end1-start)

                # else:
                #     x = self.lx[0][d][s]
                #     m = math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                #     y = 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
                #     lb[d][s] = y
                #     ub[d][s] = y
                    # coeff[0, d, s] = y / (2 * x)
                    # coeff[1, d, s] = y / (2 * m)
                    # coeff[2, d, s] = 0
                    # coeff[3, d, s] = y / (2 * x)
                    # coeff[4, d, s] = y / (2 * m)
                    # coeff[5, d, s] = 0


        # self.precal_bnd = [{}, {}, {}]
        # coeff = torch.zeros(6, size, 5 if self.method == "opt" else 1).to(dev)
        # for d in range(size):
        #     if self.method == "lp":
        #         (
        #             coeff[0, d, 0],
        #             coeff[1, d, 0],
        #             coeff[2, d, 0],
        #             coeff[3, d, 0],
        #             coeff[4, d, 0],
        #             coeff[5, d, 0],
        #         ) = LP_bounds(
        #             self.lx[d],
        #             self.ux[d],
        #             self.ly[d],
        #             self.uy[d],
        #             tanh=False,
        #             gelu=True,
        #         )

        self.precal_bnd[2] = coeff
        Al = torch.zeros(22, 1).to(dev)
        Bl = torch.zeros(22, 1).to(dev)
        Cl = torch.zeros(22, 1).to(dev)
        Au = torch.zeros(22, 1).to(dev)
        Bu = torch.zeros(22, 1).to(dev)
        Cu = torch.zeros(22, 1).to(dev)
        for k in range(22):
            for i in range(6): #
                Al[k] = Al[k]+coeff[0, k, i]
                Bl[k] = Bl[k]+coeff[1, k, i]
                Cl[k] = Cl[k]+coeff[2, k, i]
                Au[k] = Au[k]+coeff[3, k, i]
                Bu[k] = Bu[k]+coeff[4, k, i]
                Cu[k] = Cu[k]+coeff[5, k, i]
            Al[k] = Al[k]/6 #
            Bl[k] = Bl[k]/6 #
            Cl[k] = Cl[k]/6 #
            Au[k] = Au[k]/6 #
            Bu[k] = Bu[k]/6 #
            Cu[k] = Cu[k]/6 #

        for d in range(size1):
            for s in range(size2):
                if self.lx[0][d][s].item()>0:
                    # lb[d][s] = self.lx[0][d][s].item()
                    # ub[d][s] = self.ux[0][d][s].item()
                    lb[d][s] = self.gelu_new(self.lx[0][d][s].item())
                    ub[d][s] = self.gelu_new(self.ux[0][d][s].item())

                if self.ux[0][d][s].item()<=0:
                    # lb[d][s] = 0
                    # ub[d][s] = 0
                    lb[d][s] = self.gelu_new(self.lx[0][d][s].item())
                    ub[d][s] = self.gelu_new(self.ux[0][d][s].item())

                else:
                    # lb[d][s] = self.gelu_new(self.lx[0][d][s].item())
                    # ub[d][s] = self.gelu_new(self.ux[0][d][s].item())
                    lb[d][s] = Al[d].item() * self.lx[0][d][s].item() + Bl[d].item() * self.ly[0][d][s].item() + Cl[d].item()
                    ub[d][s] = Au[d].item() * self.lx[0][d][s].item() + Bu[d].item() * self.ly[0][d][s].item() + Cu[d].item()
        end3 = time.perf_counter()
        print(end3-start)






        # Al = coeff[0, :, 0]
        # Bl = coeff[1, :, 0]
        # Cl = coeff[2, :, 0]
        # Au = coeff[3, :, 0]
        # Bu = coeff[4, :, 0]
        # Cu = coeff[5, :, 0]

        # lexpr_w = torch.diag(Al) @ w_x + torch.diag(Bl) @ w_m
        # lexpr_b = torch.diag(Al) @ b_x + torch.diag(Bl) @ b_m + Cl
        # uexpr_w = torch.diag(Au) @ w_x + torch.diag(Bu) @ w_m
        # uexpr_b = torch.diag(Au) @ b_x + torch.diag(Bu) @ b_m + Cu
        #
        # lb = self.prev_layer._get_lb(torch.diag(lexpr_w), lexpr_b)
        # ub = self.prev_layer._get_ub(torch.diag(uexpr_w), uexpr_b)
        self.output_dp = DeepPoly(
            lb=lb,
            ub=ub,
            lexpr=None,
            uexpr=None,
            device=prev_dp.device,
        )

        return self.output_dp

class Conv1D(nn.Module, DPBackSubstitution):
    def __init__(self, nf, nx, prev_layer=None):
        super(Conv1D, self).__init__()
        self.prev_layer = prev_layer
        self.nf = nf
        # dp: deepPoly 多面体
        self.input_dp = None
        self.output_dp = None

    def assign(self, weight, bias, device=torch.device("cuda:0")):
        self.weight = weight.data.to(device)
        self.bias = bias.data.to(device)

    def forward(self, prev_dp):
        size_out = prev_dp.lb.size()[:-1] + (self.nf,)
        # Initial layer
        if self.prev_layer == None:
            prev_dp.device = "cuda:0"
            self.input_dp = prev_dp
            lb = (
                self.input_dp.lb.view(-1, self.input_dp.lb.size(-1)) @ positive_only(self.weight)
                + self.input_dp.ub.view(-1, self.input_dp.lb.size(-1)) @ negative_only(self.weight)
                + self.bias
            )
            # lb = (
            #     positive_only(self.weight) @ self.input_dp.lb.view(-1, self.input_dp.lb.size(-1))
            #     + negative_only(self.weight) @ self.input_dp.ub.view(-1, self.input_dp.lb.size(-1))
            #     + self.bias
            # )
            lb = lb.view(*size_out)
            ub = (
                self.input_dp.ub.view(-1, self.input_dp.lb.size(-1)) @ positive_only(self.weight)
                + self.input_dp.lb.view(-1, self.input_dp.lb.size(-1)) @ negative_only(self.weight)
                + self.bias
            )
            # ub = (
            #     positive_only(self.weight) @ self.input_dp.ub
            #     + negative_only(self.weight) @ self.input_dp.lb
            #     + self.bias
            # )
            ub = ub.view(*size_out)

        # Intermediate layer 中间层
        # 运用反向置换方法，通过前一层的神经元计算边界
        else:
            lb = self.prev_layer._get_lb(self.weight, self.bias)
            lb = lb.view(*size_out)
            ub = self.prev_layer._get_ub(self.weight, self.bias)
            ub = ub.view(*size_out)

        self.output_dp = DeepPoly(
            lb=lb,
            ub=ub,
            lexpr=(self.weight, self.bias),
            uexpr=(self.weight, self.bias),
            device=prev_dp.device,
        )

        return self.output_dp

class FeedForward(nn.Module, DPBackSubstitution):
    def __init__(self, config_resid_pdrop, block_i, blockparame, n_state=2048, nx=512, prev_layer=None): # config_resid_pdrop = 0.1
        super().__init__()
        self.c_fc = Conv1D(n_state, nx)
        self.c_fc.assign(blockparame[block_i]['mlp.c_fc.weight'], blockparame[block_i]['mlp.c_fc.bias'])
        #self.c_proj = Conv1D(nx, n_state, prev_layer=self.c_fc)
        self.c_proj = Conv1D(nx, n_state, prev_layer=None)
        self.c_proj.assign(blockparame[block_i]['mlp.c_proj.weight'], blockparame[block_i]['mlp.c_proj.bias'])
        self.act = Gelu_new(prev_layer=None)
        self.dropout = nn.Dropout(config_resid_pdrop)

    def gelu_new(self, x):
        """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
            Also see https://arxiv.org/abs/1606.08415
        """
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

    def forward(self, prev_dp):
        m = self.c_fc(prev_dp)
        n = self.act(m)
        # m.lb = self.gelu_new(m.lb)
        # m.ub = self.gelu_new(m.ub)
        # n = self.act(m)

        c_proj_out = self.c_proj(n)
        # c_proj_out = self.c_proj(self.act(self.c_fc(prev_dp)))
        lb = self.dropout(c_proj_out.lb)
        ub = self.dropout(c_proj_out.ub)

        self.output_dp = DeepPoly(
            lb=lb,
            ub=ub,
            lexpr=c_proj_out.lexpr,
            uexpr=c_proj_out.uexpr,
            device=prev_dp.device,
        )
        return self.output_dp

class Attention(nn.Module, DPBackSubstitution):
    def __init__(self, block_i, blockparame, config_resid_pdrop=0.1, d_model=512, n_head=8, n_ctx=128, scale=False, prev_layer=None, attention_mask=None):
        super().__init__()
        self.block_i = block_i
        self.blockparame = blockparame
        self.n_head = n_head
        self.d_model = d_model
        self.scale = scale
        self.attention_mask = attention_mask
        self.softmax = nn.Softmax(dim=-1)
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.dropout = nn.Dropout(config_resid_pdrop)
        self.c_attn = Conv1D(d_model * 3, d_model, prev_layer=prev_layer)
        self.c_attn.assign(blockparame[block_i]['attn.c_attn.weight'], blockparame[block_i]['attn.c_attn.bias'])
        #self.c_proj = Conv1D(d_model, d_model,prev_layer=self.c_attn)
        self.c_proj = Conv1D(d_model, d_model)
        self.c_proj.assign(blockparame[block_i]['attn.c_proj.weight'], blockparame[block_i]['attn.c_proj.bias'])

    def split_heads(self, x):
        "return shape [`batch`, `head`, `sequence`, `features`]"
        new_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def _attn(self, q, k, v, attention_mask=None):
        w = torch.matmul(q, k.transpose(-2, -1)).to(dev)
        if self.scale: w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        self.bias = self.blockparame[self.block_i]['attn.bias']
        self.masked_bias = self.blockparame[self.block_i]['attn.masked_bias']
        mask = self.bias[:, :, ns - nd : ns, :ns]
        w = torch.where(mask.bool(), w, self.masked_bias.to(w.dtype))

        if self.attention_mask is not None:
            w = w + self.attention_mask.to(dev)
        w = self.softmax(w)
        w = self.dropout(w)
        outputs = torch.matmul(w, v.to(dev))
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_shape)

    def forward(self, prev_dp):
        c_attn_out = self.c_attn(prev_dp)  # new `x` shape - `[1,3,2304]`
        q_lb, k_lb, v_lb = c_attn_out.lb.split(self.d_model, dim=2)
        q_ub, k_ub, v_ub = c_attn_out.ub.split(self.d_model, dim=2)
        q_lb, k_lb, v_lb = self.split_heads(q_lb), self.split_heads(k_lb), self.split_heads(v_lb)
        q_ub, k_ub, v_ub = self.split_heads(q_ub), self.split_heads(k_ub), self.split_heads(v_ub)
        out_lb = self._attn(q_lb, k_lb, v_lb)
        out_lb = self.merge_heads(out_lb)
        out_ub = self._attn(q_ub, k_ub, v_ub)
        out_ub = self.merge_heads(out_ub)
        out = DeepPoly(
            lb=out_lb,
            ub=out_ub,
            lexpr=c_attn_out.lexpr,
            uexpr=c_attn_out.uexpr,
            device=prev_dp.device,
        )
        out = self.c_proj(out)

        self.output_dp = DeepPoly(
            lb=out.lb,
            ub=out.ub,
            lexpr=out.lexpr,
            uexpr=out.uexpr,
            device=prev_dp.device,
        )
        return self.output_dp

class LayerNorm(nn.LayerNorm, DPBackSubstitution):
    def __init__(self, normalized_shape):
        super(LayerNorm, self).__init__(normalized_shape)
        self.input_dp = None
        self.output_dp = None
        self.layer = nn.LayerNorm(normalized_shape).to(dev)

    def forward(self, prev_dp):
        lb = self.layer(prev_dp.lb)
        ub = self.layer(prev_dp.ub)

        self.output_dp = DeepPoly(
            lb=lb,
            ub=ub,
            lexpr=prev_dp.lexpr,
            uexpr=prev_dp.uexpr,
            device=prev_dp.device,
        )

        return self.output_dp

class TransformerBlock(nn.Module, DPBackSubstitution):
    def __init__(self, config_resid_pdrop, block_i, blockparame, d_model=512, n_head=12, dropout=0.1, attention_mask=None, prev_layer=None):
        super(TransformerBlock, self).__init__()
        self.ln_1 = LayerNorm(d_model)
        # self.ln_1.assign(blockparame[block_i]['ln_1.weight'], blockparame[block_i]['ln_1.bias'])
        self.attn = Attention(block_i, blockparame, config_resid_pdrop, d_model=512, n_head=8, n_ctx=128, scale=False, prev_layer=prev_layer, attention_mask=attention_mask)
        self.ln_2 = LayerNorm(d_model)
        # self.ln_2.assign(blockparame[block_i]['ln_2.weight'], blockparame[block_i]['ln_2.bias'])
        self.feedforward = FeedForward(config_resid_pdrop, block_i, blockparame, n_state=2048, nx=512, prev_layer=self.attn.c_proj)

    def forward(self, prev_dp):
        ln_1_out = self.ln_1(prev_dp)
        attn_out = self.attn(ln_1_out)
        attn_out.lb = prev_dp.lb + attn_out.lb
        attn_out.ub = prev_dp.ub + attn_out.ub
        ln_2_out = self.ln_2(attn_out)
        feedforward_out = self.feedforward(ln_2_out)
        feedforward_out.lb = attn_out.lb + feedforward_out.lb
        feedforward_out.ub = attn_out.ub + feedforward_out.ub

        return feedforward_out

class GPT2modeldp(nn.Module, DPBackSubstitution):
    def __init__(self, prev_layer=None, pos_ids=None ,config_resid_pdrop=0.1, attention_mask=None):
        super(GPT2modeldp, self).__init__()
        self.prev_layer = prev_layer
        self.output_dp = None
        self.pos_ids = pos_ids
        self.config_resid_pdrop = config_resid_pdrop
        self.attention_mask = attention_mask

    def get_state_dict(self, model_path):
        # model_path = '/home/dhd/DRLVerification/Transformer4NetworkTraffic/gpt_model/classifier/GPT2Classifier_checkpoints/epoch=11-val_loss=0.04-other_metric=0.00.ckpt'

        checkpoint = torch.load(model_path)
        # odict_keys(['gpt2.wte.weight', 'gpt2.wpe.weight', ..., 'gpt2.ln_f.weight', 'gpt2.ln_f.bias', 'fc.weight', 'fc.bias'])
        paramelist = ['ln_1.weight', 'ln_1.bias', 'attn.bias', 'attn.masked_bias', 'attn.c_attn.weight',
                      'attn.c_attn.bias',
                      'attn.c_proj.weight', 'attn.c_proj.bias', 'ln_2.weight', 'ln_2.bias', 'mlp.c_fc.weight',
                      'mlp.c_fc.bias',
                      'mlp.c_proj.weight', 'mlp.c_proj.bias']
        blockparame = {}
        for i in range(6):#
            blockparame[i] = {}
            block = 'gpt2.h.' + str(i) + '.'
            for parame in paramelist:
                parameters = block + parame
                blockparame[i][parame] = checkpoint["state_dict"][parameters]

        return checkpoint, blockparame

    def forward(self, prev_dp):
        dev = prev_dp.device
        print("the devdev:" + str(dev))
        model_path = '/home/dhd/DRLVerification/Transformer4NetworkTraffic/gpt_model/classifier/GPT2Classifier_checkpoints/epoch=11-val_loss=0.04-other_metric=0.00.ckpt' #
        checkpoint, blockparame = self.get_state_dict(model_path)
        wte = nn.Embedding(10035, 512, _weight=checkpoint["state_dict"]['gpt2.wte.weight']).to(dev)
        wpe = nn.Embedding(128, 512, _weight=checkpoint["state_dict"]['gpt2.wpe.weight']).to(dev)
        # wte = Linear(10035, 512, prev_layer=None).to(dev)
        # wte.assign(checkpoint["state_dict"]['gpt2.wte.weight'])
        # wpe = Linear(128, 512, prev_layer=wte).to(dev)
        # wpe.assign(checkpoint["state_dict"]['gpt2.wpe.weight'])

        if self.pos_ids is None:
            self.pos_ids = torch.arange(0, prev_dp.lb.size(-1)).unsqueeze(0).to(dev)
        wte_out_lb = wte(prev_dp.lb)
        wte_out_ub = wte(prev_dp.ub)
        wpe_out = wpe(self.pos_ids)
        hiddenstateslb = wte_out_lb + wpe_out
        hiddenstatesub = wte_out_ub + wpe_out
        block_out = DeepPoly(
            lb=hiddenstateslb,
            ub=hiddenstatesub,
            lexpr=prev_dp.lexpr,
            uexpr=prev_dp.uexpr,
            device=prev_dp.device,
        )
        # block = TransformerBlock(self.config_resid_pdrop, 5, blockparame, d_model=512, n_head=12, dropout=0.1,
        #                          attention_mask=self.attention_mask)
        # block_out = block(block_out)
        for i in range(6):#
            block = TransformerBlock(self.config_resid_pdrop, i, blockparame, d_model=512, n_head=12, dropout=0.1, attention_mask=self.attention_mask)
            block_out = block(block_out)

        self.ln_f = LayerNorm(512)
        # self.ln_f.assign(checkpoint["state_dict"]['gpt2.ln_f.weight'], checkpoint["state_dict"]['gpt2.ln_f.bias'])
        ln_f_out = self.ln_f(block_out)

        return ln_f_out

class Sigmoidal(nn.Sigmoid, DPBackSubstitution):
    def __init__(self, func, prev_layer=None):
        if func in ["sigmoid", "tanh"]:
            super(Sigmoidal, self).__init__()
            self.func = func
        else:
            raise RuntimeError("not supported sigmoidal layer")
        self.prev_layer = prev_layer
        self.output_dp = None

    def forward(self, prev_dp):
        dim = prev_dp.dim
        dev = prev_dp.device
        lexpr_w = torch.zeros(dim).to(device=dev)
        lexpr_b = torch.zeros(dim).to(device=dev)
        uexpr_w = torch.zeros(dim).to(device=dev)
        uexpr_b = torch.zeros(dim).to(device=dev)

        # intermediate layer
        for i in range(dim):
            l, u = prev_dp.lb[i], prev_dp.ub[i]
            if self.func == "sigmoid":
                sl, su = torch.sigmoid(l), torch.sigmoid(u)
                lmb = (su - sl) / (u - l) if l < u else sl * (1 - sl)
                lmb_ = torch.min(su * (1 - su), sl * (1 - sl))
            elif self.func == "tanh":
                sl, su = torch.tanh(l), torch.tanh(u)
                lmb = (su - sl) / (u - l) if l < u else 1 - sl * sl
                lmb_ = torch.min(1 - su * su, 1 - sl * sl)
            if l > 0:
                lexpr_w[i] = lmb
                lexpr_b[i] = sl - lmb * l
            else:
                lexpr_w[i] = lmb_
                lexpr_b[i] = sl - lmb_ * l
            if u < 0:
                uexpr_w[i] = lmb
                uexpr_b[i] = su - lmb * u
            else:
                uexpr_w[i] = lmb_
                uexpr_b[i] = su - lmb_ * u

        lb = self.prev_layer._get_lb(torch.diag(lexpr_w), lexpr_b)
        ub = self.prev_layer._get_ub(torch.diag(uexpr_w), uexpr_b)
        self.output_dp = DeepPoly(
            lb=lb,
            ub=ub,
            lexpr=(lexpr_w, lexpr_b),
            uexpr=(uexpr_w, uexpr_b),
            device=prev_dp.device,
        )

        return self.output_dp
