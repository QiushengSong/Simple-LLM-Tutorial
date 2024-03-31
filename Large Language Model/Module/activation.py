import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def Sigmoid(x):
    """
    Sigmoid Activation Function

    Args:
        x(torch.Tensor): input tensor
    """
    return 1 / (1 + torch.exp(-x))


def Softmax(x):
    """
    Softmax Activation Function

    Args:
        x(torch.Tensor): input tensor
    """
    exp_x = torch.exp(x)
    softmax_x = exp_x / torch.sum(exp_x, dim=1, keepdim=True)
    return softmax_x


def ReLU(x):
    """
    Rectified Linear Unit Activation Function
    """
    return max(x, 0)


def Tanh(x):
    """
    Hyperbolic Tangent Activation Function
    """
    return (2 / (1 + torch.exp(-2 * x))) - 1


def LeakyReLU(x,
              gamma: float = 0.1):
    """
    Leaky ReLU Activation Function
    """
    return max(0, x) + gamma * min(0, x)


def GeLU(x):
    """
    Gaussian Error Linear Unit
    """
    return x * (1 + torch.tanh(torch.sqrt(torch.tensor(2)/torch.pi) * (x + 0.044715*x**3)))


class ParametericReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):

        return max(0, x) + self.gamma * min(0, x)


class ELU(nn.Module):
    """
    Exponential Linear Unit Activation Function
    """
    def __init__(self,
                 alpha: float
                 ):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return torch.where(x > 0, x, self.alpha*(torch.exp(x) - 1))


class SeLU(nn.Module):
    """
    Scaled ELU Activation Function
    """
    def __init__(self,
                 factor: float = 1.05070098,
                 alpha: float = 1.67326324,
                 ):
        super().__init__()
        self.factor = factor
        self.alpha = alpha

    def forward(self, x):
        value = self.factor * torch.where(x > 0, x, self.alpha * (torch.exp(x) - 1))
        return value


class Swish(nn.Module):
    """
        Swish Activation Function

        Args:
            beta(float): a trainable parameter that controls the smoothness of the sigmoid function
        """
    def __init__(self,
                 beta=1.0
                 ):
        super().__init__()

        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class GLU(nn.Module):
    """
    Gating Linear Unit Activation Function
    """
    def __init__(self,
                 in_feature: int,
                 out_feature: int,
                 ):
        super().__init__()
        self.gate_weight = nn.Parameter(torch.ones((in_feature, out_feature)))
        self.linear_weight = nn.Parameter(torch.ones((in_feature, out_feature)))
        self.gate_bias = nn.Parameter(torch.zeros(out_feature))
        self.linear_bias = nn.Parameter(torch.zeros(out_feature))

    def forward(self, x):
        gate = torch.sigmoid(F.linear(input=x,
                                      weight=self.gate_weight,
                                      bias=self.gate_bias))
        linear_output = F.linear(input=x,
                                 weight=self.linear_weight,
                                 bias=self.linear_bias)
        return gate * linear_output


class ReGLU(nn.Module):
    """
    ReGLU Activation Function
    """

    def __init__(self,
                 in_feature: int,
                 out_feature: int,
                 ):
        super().__init__()
        self.gate_weight = nn.Parameter(torch.ones((in_feature, out_feature)))
        self.linear_weight = nn.Parameter(torch.ones((in_feature, out_feature)))
        self.gate_bias = nn.Parameter(torch.zeros(out_feature))
        self.linear_bias = nn.Parameter(torch.zeros(out_feature))

    def forward(self, x):
        gate = F.relu(F.linear(input=x,
                               weight=self.gate_weight,
                               bias=self.gate_bias))
        linear_output = F.linear(input=x,
                                 weight=self.linear_weight,
                                 bias=self.linear_bias)
        return gate * linear_output


class SwiGLU(nn.Module):
    """
    SwiGLU Activation Function

    Args:
        beta(float): a trainable parameter that controls the smoothness of the sigmoid function
    """

    def __init__(self,
                 in_feature: int,
                 out_feature: int,
                 beta=1.0,
                 ):
        super().__init__()
        self.gate_weight = nn.Parameter(torch.ones((in_feature, out_feature)))
        self.linear_weight = nn.Parameter(torch.ones((in_feature, out_feature)))
        self.gate_bias = nn.Parameter(torch.zeros(out_feature))
        self.linear_bias = nn.Parameter(torch.zeros(out_feature))
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, x):

        swish = x * torch.sigmoid(self.beta * x)
        gate = swish * (F.linear(input=x,
                                 weight=self.gate_weight,
                                 bias=self.gate_bias))
        linear_output = F.linear(input=x,
                                 weight=self.linear_weight,
                                 bias=self.linear_bias)
        return gate * linear_output
