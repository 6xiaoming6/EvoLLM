import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.rms_norm import RMSNorm
from models.modules.attention import Attention
from models.modules.feed_forward import FeedForward
from configs.evollm_config import EvoLLMConfig

class EvoLLMBlock(nn.Module):
    def __init__(self, cfg: EvoLLMConfig):
        super().__init__()
        self.input_norm = RMSNorm(cfg)
        self.attention = Attention(cfg)
        self.post_attention_norm = RMSNorm(cfg)
        self.ffn = FeedForward(cfg)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)
        hidden_states = self.attention(hidden_states) + residual

        residual = hidden_states
        hidden_states = self.post_attention_norm(hidden_states)
        output = self.ffn(hidden_states) + residual

        return output 

