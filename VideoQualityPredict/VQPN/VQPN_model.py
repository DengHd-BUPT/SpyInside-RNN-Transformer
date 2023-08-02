import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import time
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class VQPNModel(nn.Module):
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
        self.sigmoid = nn.Sigmoid().to(dev)

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
        gru_result = self.lin2(gru_out[1][0])
        gru_result = self.relu(gru_result)
        out = self.lin3(gru_result)
        out = self.sigmoid(out)
        return out
