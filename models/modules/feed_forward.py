import torch
import torch.nn as nn
from configs.evollm_config import EvoLLMConfig

class FeedForward(nn.Module):
    def __init__(self, cfg: EvoLLMConfig):
        super().__init__()
        self.intermediate_size = cfg.intermediate_size
        self.hidden_size = cfg.hidden_size

        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=cfg.use_bias)
        self.act_fn = nn.GELU()
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=cfg.use_bias)
        self.dropout = nn.Dropout(cfg.dropout_ratio)

    def forward(self, x):
        x = self.up_proj(x)
        x = self.act_fn(x)
        x = self.down_proj(x)
        
        return self.dropout(x)
