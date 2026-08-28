import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.evollm_config import EvoLLMConfig

class Attention(nn.Module):
    def __init__(self, cfg: EvoLLMConfig):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.head_dim

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=cfg.use_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=cfg.use_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=cfg.use_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=cfg.use_bias)
        self.attn_dropout = nn.Dropout(cfg.dropout_ratio)
        self.resid_dropout = nn.Dropout(cfg.dropout_ratio)

    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape

        # (batch_size, seq_len, num_head * head_dim)
        q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        # (batch_size, num_head, seq_len, head_dim)
        q, k, v = (
            q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2),
            k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2),
            v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        )
        causal_mask = torch.full((seq_len, seq_len), float("-inf"), device=hidden_states.device).triu(diagonal=1)

        scores = (q @ k.transpose(1, 2)) / math.sqrt(self.head_dim)
        scores += causal_mask
        # (batch_size, num_head, seq_len, head_dim)
        output = self.attn_dropout(F.softmax(scores, dim=-1)) @ v
        # (batch_size, seq_len, num_head * head_dim)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)

        output = self.o_proj(output)
        output = self.resid_dropout(output)
        return output

