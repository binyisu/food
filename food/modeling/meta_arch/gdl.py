import torch
import torch.nn as nn
from torch.autograd import Function


class GradientDecoupleLayer(Function):

    @staticmethod
    def forward(ctx, x, _lambda):
        ctx._lambda = _lambda
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output * ctx._lambda
        return grad_output, None


class AffineLayer(nn.Module):
    def __init__(self, num_channels, bias=False):
        super(AffineLayer, self).__init__()
        weight = torch.FloatTensor(1, num_channels, 1, 1).fill_(1) # [1, 1024, 1, 1]
        self.weight = nn.Parameter(weight, requires_grad=True) # [1, 1024, 1, 1]

        self.bias = None
        if bias:
            bias = torch.FloatTensor(1, num_channels, 1, 1).fill_(0) # [1, 1024, 1, 1]
            self.bias = nn.Parameter(bias, requires_grad=True) # [1, 1024, 1, 1]

    def forward(self, X): # [8, 1024, 32, 32]
        out = X * self.weight.expand_as(X) # pixel wise multiply
        if self.bias is not None:
            out = out + self.bias.expand_as(X) # y=wx+b
        return out # [8, 1024, 32, 32]


def decouple_layer(x, _lambda):
    return GradientDecoupleLayer.apply(x, _lambda)
