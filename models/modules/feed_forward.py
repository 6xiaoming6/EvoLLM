import torch
import torch.nn as nn
from configs.evollm_config import EvoLLMConfig
from transformers.activations import ACT2FN

class FeedForward(nn.Module):
    def __init__(self, cfg: EvoLLMConfig):
        super().__init__()
        self.intermediate_size = cfg.intermediate_size
        self.hidden_size = cfg.hidden_size

        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=cfg.use_bias)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=cfg.use_bias)
        self.silu = nn.SiLU() # Swish激活函数
        
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=cfg.use_bias)
        self.dropout = nn.Dropout(cfg.dropout_ratio)

    def forward(self, x):
        # SwiGlu = silu(x*w1) * x*w2
        x = self.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        
        return self.dropout(x)
