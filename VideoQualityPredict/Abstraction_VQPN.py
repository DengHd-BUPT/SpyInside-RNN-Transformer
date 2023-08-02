import sys
sys.path.append("..")
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from utils import get_default_device
import Abstraction_RNN as FormalNN
import time

dev = get_default_device()

class QARCModel(nn.Module):
    def __init__(
        self,
        kernel,
        dense_size,
        input_w = 64,
        input_h = 36,
        input_d = 3,
        input_seq = 25,
        output_dim = 5,
    ):
        self.input_w = input_w
        self.input_h = input_h
        self.input_d = input_d
        self.input_seq = input_seq
        self.dense_size = dense_size

        super().__init__()
        self.conv1 = nn.Conv2d(input_d, kernel, kernel_size = 5, padding= 'same').to(dev)
        self.relu = nn.ReLU().to(dev)
        self.avgpool = nn.AvgPool2d(3, padding = 1).to(dev)
        self.conv2 = nn.Conv2d(kernel, kernel, kernel_size = 3, padding = 'same').to(dev)
        self.maxcpool = nn.MaxPool2d(2).to(dev)
        self.lin1 = nn.Linear(4224, dense_size).to(dev)
        self.relu = nn.ReLU().to(dev)
        self.gru1 = nn.GRU(dense_size, dense_size, batch_first=True).to(dev)
        self.gru2 = nn.GRU(dense_size, dense_size, batch_first=True, dropout = 0.8).to(dev)
        self.lin2 = nn.Linear(dense_size, dense_size).to(dev)
        self.lin3 = nn.Linear(dense_size, output_dim).to(dev)
        self.sigmoid = nn.Sigmoid()

    def CNNCore(self, x):
        net = self.conv1(x)
        net = self.relu(net)
        net = self.avgpool(net)
        net = self.conv2(net)
        net = self.relu(net)
        net = self.maxcpool(net)
        net = torch.flatten(net, start_dim = 1)
        net = self.lin1(net)
        net = self.relu(net)
        split_net = torch.flatten(net, start_dim = 1)
        return split_net

    def forward(self, input):
        _split_array = []
        input = input.float()
        for i in range(self.input_seq):
            tmp_network = input[:, i:i+1, :, :, :].reshape(-1, self.input_d, self.input_h, self.input_w)
            _split_array.append(self.CNNCore(tmp_network))
        merge_net = torch.cat(_split_array, 1)
        merge_net = torch.flatten(merge_net, start_dim=1)
        _count = list(merge_net.size())[1]
        net = merge_net.reshape(-1, int(_count / self.dense_size), self.dense_size)
        net = self.gru1(net)
        gru_out = self.gru2(net[0])
        gru_out = self.lin2(gru_out[1][0])
        gru_result = self.relu(gru_out)
        out = self.lin3(gru_result)
        out = self.sigmoid(out)
        # out = self.linear3(gru_result)
        return out

class QARCModelDP(QARCModel):
    def __init__(self, *args, **kwargs):
        super(QARCModelDP, self).__init__(*args, **kwargs)

    def set_bound_method(self, bound_method):
        self.bound_method = bound_method

    def certify(self, input, eps):
        layers = []
        gru_pack = []
        feed = []
        _split_arraylb = []
        _split_arrayub = []
        dev = input.device

        for i in range(self.input_seq):
            tmp_input = input[:, i:i+1, :, :, :].reshape(-1, self.input_d, self.input_h, self.input_w)
            tmp_input = tmp_input.float()
            tmp_input = FormalNN.DeepPoly.deeppoly_from_perturbation(tmp_input, eps, truncate=(0, 1))
            net = FormalNN.DeepPoly(
                torch.t(tmp_input.lb),
                torch.t(tmp_input.ub),
                None,
                None,
            )
            chain = []
            conv1 = FormalNN.Conv2d.convert(self.conv1)
            conv1out = conv1(net)
            avgpool = FormalNN.AvgPool2d(3)
            avgpoolout = avgpool.forward(conv1out)
            maxpool = FormalNN.MaxPool2d(2)
            maxpoolout = maxpool.forward(avgpoolout)
            net = maxpoolout
            net.lb = torch.flatten(maxpoolout.lb, start_dim=1)
            net.ub = torch.flatten(maxpoolout.ub, start_dim=1)
            lin1 = FormalNN.Linear.convert(self.lin1, device=dev)
            lin1_out = lin1(net)
            chain.append(lin1)
            lin1_out.lb = torch.flatten(lin1_out.lb, start_dim=1)
            lin1_out.ub = torch.flatten(lin1_out.ub, start_dim=1)
            _split_arraylb.append(lin1_out.lb)
            _split_arrayub.append(lin1_out.ub)

        merge_netlb = torch.cat(_split_arraylb, 1)
        merge_netlb = torch.flatten(merge_netlb, start_dim=1)
        merge_netub = torch.cat(_split_arrayub, 1)
        merge_netub = torch.flatten(merge_netub, start_dim=1)
        _count = list(merge_netlb.size())[1]
        cnn_outlb = merge_netlb.reshape(-1, int(_count / self.dense_size), self.dense_size)
        cnn_outub = merge_netub.reshape(-1, int(_count / self.dense_size), self.dense_size)
        cnn_out = FormalNN.DeepPoly(
            cnn_outlb,
            cnn_outub,
            lin1_out.lexpr,
            lin1_out.uexpr,
        )
        gru1 = FormalNN.GRUCell.convert(
            self.gru1,
            prev_layer=lin1,
            prev_cell=None,
            method=self.bound_method,
        )
        gru1_out = gru1(cnn_out)
        chain.append(gru1)
        gru2 = FormalNN.GRUCell.convert(
            self.gru2,
            prev_layer=gru1,
            prev_cell=None,
            method=self.bound_method,
        )
        gru2_out = gru2(gru1_out)
        chain.append(gru2)

        lin2 = FormalNN.Linear.convert(self.lin2, prev_layer=gru2, device=dev)
        lin2_out=lin2(gru2_out)
        chain.append(lin2)
        lin2_out.lb = self.relu(lin2_out.lb)
        lin2_out.ub = self.relu(lin2_out.ub)
        layers.append(chain)
        lin3 = FormalNN.Linear.convert(self.lin3, prev_layer=lin2, device=dev)
        lin3_out = lin3(lin2_out)
        chain.append(lin3)
        sigmoid = FormalNN.Sigmoidal("sigmoid", prev_layer=lin3)
        sigmoid_out = sigmoid(lin3_out)
        chain.append(sigmoid)
        layers.append(sigmoid)
        print("Last sigmoid_out: " + str(sigmoid_out))
        return sigmoid_out

