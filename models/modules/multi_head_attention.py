import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import repeat_kv, apply_rope
from models.modules.rms_norm import RMSNorm
from configs.evollm_config import EvoLLMConfig


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: EvoLLMConfig):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.num_attention_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_kv_heads
        assert self.num_attention_heads % self.num_kv_heads == 0
        self.repeat_num = self.num_attention_heads // self.num_kv_heads
        self.head_dim = cfg.head_dim
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=cfg.use_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=cfg.use_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=cfg.use_bias)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=cfg.use_bias)
        self.attn_dropout = nn.Dropout(cfg.dropout_ratio)
        self.resid_dropout = nn.Dropout(cfg.dropout_ratio)



    def forward(self, hidden_states, position_embeddings, past_key_value = None, use_cache = False):
        batch_size, seq_len, _ = hidden_states.shape

        # q:    (batch_size, seq_len, num_attention_heads * head_dim)
        # k, v: (batch_size, seq_len, num_kv_heads * head_dim)
        q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)

        # q:    (batch_size, num_attention_heads, seq_len, head_dim)
        # k, v: (batch_size, num_kv_heads, seq_len, head_dim)
        q, k, v = (
            q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2),
            k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2),
            v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        )
        freqs_sin, freqs_cos = position_embeddings
        q, k = apply_rope(q, freqs_sin, freqs_cos), apply_rope(k, freqs_sin, freqs_cos)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        present_key_value = (k, v) if use_cache else None

        # (batch_size, num_attention_heads, seq_len, head_dim)
        k, v = repeat_kv(k, self.repeat_num), repeat_kv(v, self.repeat_num)

        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if self.is_causal:
            causal_mask = torch.full((seq_len, seq_len), float("-inf"), device=hidden_states.device).triu(diagonal=1)
            scores[:, :, :, -seq_len: ] += causal_mask

        # (batch_size, num_attention_heads, seq_len, head_dim)
        output = self.attn_dropout(F.softmax(scores, dim=-1)) @ v
        # (batch_size, seq_len, num_attention_heads * head_dim)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)

        output = self.o_proj(output)
        output = self.resid_dropout(output)
        return output, present_key_value
