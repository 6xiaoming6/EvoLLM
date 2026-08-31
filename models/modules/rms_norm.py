import torch
import torch.nn as nn
from configs.evollm_config import EvoLLMConfig

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(self.dim))

    def forward(self, x):
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps) # x = x/根号下(variance + eps)
        return x * self.weights