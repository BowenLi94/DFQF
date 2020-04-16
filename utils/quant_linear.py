import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from functools import reduce, partial
from enum import Enum

#from utils.q_utils import *
import utils.clip 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd




##########################################################
class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = torch.round(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def quantize(tensor, num_bits, alpha, signed=True, stochastic = False):
    assert alpha > 0
    num_bins = 2 ** num_bits
    if signed:
        qmax = 2 ** (num_bits - 1) - 1
        qmin =-2 ** (num_bits - 1)
    else:
        qmax = 2 ** num_bits - 1
        qmin = 0 

    delta = alpha / qmax
    delta = max(delta, 1e-8)
    #print(delta)

    # quantize
    t_q = tensor / delta

    # stochastic rounding
    if stochastic:
        with torch.no_grad():
            noise = t_q.new_empty(t_q.shape).uniform_(-0.5, 0.5)
            t_q += noise

    # clamp and round
    t_q = torch.clamp(t_q, qmin, qmax)
    t_q = RoundSTE.apply(t_q)
    assert torch.unique(t_q).shape[0] <= num_bins

    # de-quantize
    t_q = t_q * delta
    return t_q


class ReLU_DFQF(nn.Module):
    def __init__(self, module, num_bits, init_act_clip_val ):
        super(ReLU_DFQF, self).__init__()
        self.module = module
        self.num_bits = num_bits
        self.clip_a = nn.Parameter(torch.Tensor([init_act_clip_val]))
        #self.register_buffer('clip_a', torch.Tensor([init_act_clip_val]))

    def forward(self, input):
        # Clip between 0 to the learned clip_val
        input = self.module(input)
        return quantize(input, self.num_bits, self.clip_a, signed=False) 


class Conv2d_DFQF(nn.Conv2d):
    def __init__(self, module, num_bits, init_w_clip_val, stochastic=False):
        super(Conv2d_DFQF, self).__init__(module.in_channels, module.out_channels, module.kernel_size,
                            module.stride,module.padding, module.dilation, module.groups, module.bias is not None)
        self.num_bits = num_bits
        #self.module = module
        self.clip_w = nn.Parameter(torch.Tensor([init_w_clip_val]))
        #self.register_buffer('clip_w', torch.Tensor([init_w_clip_val]))
        self.stochastic = stochastic

        
        setattr(self, 'weight', module.weight)
        setattr(self, 'bias', module.bias)


    def forward(self, input):

        weight_q = quantize(self.weight, self.num_bits, self.clip_w)
        return F.conv2d(input, weight_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)        

class Linear_DFQF(nn.Linear):
    def __init__(self, module, num_bits, init_w_clip_val, stochastic=False):
        super(Linear_DFQF, self).__init__(module.in_features, module.out_features, module.bias is not None)

        self.num_bits = num_bits
        #self.module = module
        self.clip_w = nn.Parameter(torch.Tensor([init_w_clip_val]))
        #self.register_buffer('clip_w', torch.Tensor([init_w_clip_val]))
        self.stochastic = stochastic

        
        setattr(self, 'weight', module.weight)
        setattr(self, 'bias', module.bias)
        #delattr(module, 'weight')
        #delattr(module, 'bias')

    def forward(self, input):

        weight_q = quantize(self.weight, self.num_bits, self.clip_w)
        return F.linear(input, weight_q, self.bias)
