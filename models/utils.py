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
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def nuclear_norm(B, iter, frame):
    loss_nuc = 0
    try:
        loss_nuc += torch.norm(torch.squeeze(B)[:, :, int(iter % frame)].to(device), 'nuc').to(device)
    except:
        loss_nuc += torch.norm(torch.squeeze(B)[:, :, int((iter + 1) % frame)].to(device), 'nuc').to(device)
    return loss_nuc


def Segmentation(img):
    maxvalue = np.max(img)
    thresh = 0.4 * maxvalue
    img[img >= thresh] = 255
    img[img < thresh] = 0
    return img


class soft(nn.Module):
    def __init__(self):
        super(soft, self).__init__()

    def forward(self, x, lam):
        x_abs = x.abs() - lam
        zeros = x_abs - x_abs
        n_sub = torch.max(x_abs, zeros)
        x_out = torch.mul(torch.sign(x), n_sub)
        return x_out



