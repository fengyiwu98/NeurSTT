"""
NeurSTT: Neural Spatial-Temporal Tensor Representation for Infrared Small Target Detection

This code is part of the official implementation of the paper:
"Neural Spatial-Temporal Tensor Representation for Infrared Small Target Detection"

Paper Source: https://arxiv.org/abs/2412.17302
Authors: Fengyi Wu, Simin Liu, Haoan Wang, Bingjie Tao, Junhai Luo, Zhenming Peng

Contact Information:
Fengyi Wu
Email: wufengyi98@163.com
"""
import torch
from torch import nn, optim
dtype = torch.FloatTensor
import math


class NeurSTT(nn.Module):
    def __init__(self, r_1, r_2, r_3, mid_channel):
        super(NeurSTT, self).__init__()
        # Initialize parameters
        self.centre = nn.Parameter(torch.Tensor(r_1, r_2, r_3).type(dtype))
        self.U_net1 = nn.Parameter(torch.Tensor(mid_channel, 1).type(dtype))
        self.U_net2 = nn.Parameter(torch.Tensor(mid_channel, mid_channel).type(dtype))
        self.U_net3 = nn.Parameter(torch.Tensor(r_1, mid_channel).type(dtype))
        self.V_net1 = nn.Parameter(torch.Tensor(mid_channel, 1).type(dtype))
        self.V_net2 = nn.Parameter(torch.Tensor(mid_channel, mid_channel).type(dtype))
        self.V_net3 = nn.Parameter(torch.Tensor(r_2, mid_channel).type(dtype))
        self.W_net1 = nn.Parameter(torch.Tensor(mid_channel, 1).type(dtype))
        self.W_net2 = nn.Parameter(torch.Tensor(mid_channel, mid_channel).type(dtype))
        self.W_net3 = nn.Parameter(torch.Tensor(r_3, mid_channel).type(dtype))
        # Initialize weights
        self.init_weights()

    def init_weights(self):
        stdv = 1 / math.sqrt(self.centre.size(0))
        self.centre.data.uniform_(-stdv, stdv)
        std = 5
        for net in [self.U_net1, self.U_net2, self.U_net3, self.V_net1, self.V_net2, self.V_net3, self.W_net1, self.W_net2, self.W_net3]:
            nn.init.kaiming_normal_(net, a=math.sqrt(std))

    def forward(self, U_input, V_input, W_input, U_input_tv, V_input_tv, W_input_tv, omega):
        U = self.U_net3 @ torch.sin(omega * self.U_net2 @ torch.sin(omega * self.U_net1 @ U_input))  # r_1 n_1
        V = self.V_net3 @ torch.sin(omega * self.V_net2 @ torch.sin(omega * self.V_net1 @ V_input))  # r_2 n_2
        W = self.W_net3 @ torch.sin(omega * self.W_net2 @ torch.sin(omega * self.W_net1 @ W_input))  # r_3 n_3

        U_tv = self.U_net3 @ torch.sin(omega * self.U_net2 @ torch.sin(omega * self.U_net1 @ U_input_tv))  # r_1 n_1
        V_tv = self.V_net3 @ torch.sin(omega * self.V_net2 @ torch.sin(omega * self.V_net1 @ V_input_tv))  # r_2 n_2
        W_tv = self.W_net3 @ torch.sin(omega * self.W_net2 @ torch.sin(omega * self.W_net1 @ W_input_tv))  # r_3 n_3

        out = self.centre.permute(1, 2, 0)  # r2 r3 r1
        out = out @ U  # r2 r3 n1

        out = out.permute(1, 2, 0)  # r3 n1 r2
        out = out @ V  # r3 n1 n2

        out = out.permute(1, 2, 0)  # n1 n2 r3
        out = out @ W  # n1 n2 n3

        out_tv = self.centre.permute(1, 2, 0)  # r2 r3 r1
        out_tv = out_tv @ U_tv  # r2 r3 n1

        out_tv = out_tv.permute(1, 2, 0)  # r3 n1 r2
        out_tv = out_tv @ V_tv  # r3 n1 n2

        out_tv = out_tv.permute(1, 2, 0)  # n1 n2 r3
        out_tv = out_tv @ W_tv  # n1 n2 n3

        dx_leaf = self.U_net3 @ (
                    omega * torch.cos(omega * self.U_net2 @ torch.sin(omega * self.U_net1 @ U_input_tv)) * (
                        self.U_net2 @ (
                        omega * self.U_net1.repeat(1, U_input_tv.shape[1]) * torch.cos(
                    omega * self.U_net1 @ U_input_tv))))
        dy_leaf = self.V_net3 @ (
                    omega * torch.cos(omega * self.V_net2 @ torch.sin(omega * self.V_net1 @ V_input_tv)) * (
                        self.V_net2 @ (
                        omega * self.V_net1.repeat(1, V_input_tv.shape[1]) * torch.cos(
                    omega * self.V_net1 @ V_input_tv))))
        dz_leaf = self.W_net3 @ (
                    omega * torch.cos(omega * self.W_net2 @ torch.sin(omega * self.W_net1 @ W_input_tv)) * (
                        self.W_net2 @ (
                        omega * self.W_net1.repeat(1, W_input_tv.shape[1]) * torch.cos(
                    omega * self.W_net1 @ W_input_tv))))

        dx = (((self.centre @ W_tv).permute(2, 0, 1) @ V_tv).permute(0, 2, 1) @ dx_leaf).permute(2, 1, 0)
        dy = (((self.centre @ W_tv).permute(1, 2, 0) @ U_tv).permute(1, 2, 0) @ dy_leaf).permute(1, 2, 0)
        dz = (((self.centre.permute(1, 2, 0) @ U_tv).permute(1, 2, 0) @ V_tv).permute(1, 2, 0) @ dz_leaf)


        return out, out_tv, dx, dy, dz