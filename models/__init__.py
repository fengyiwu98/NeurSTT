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
from .model_NeurSTT import *

def get_model(name, r_1, r_2, r_3, mid):
    if name == 'NeurSTT':
        net = NeurSTT(r_1, r_2, r_3, mid)

    else:
        raise NotImplementedError

    return net
