import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import repeat_kv
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



    def forward(self, hidden_states, past_key_value = None, use_cache = False):
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

# 训练中走完完整的压缩和解压缩，推理时进行矩阵合并
# =========================
# Query低秩压缩
#
# hidden_size
#     ↓
# q_lora_rank
#     ↓
# num_heads * head_dim
# =========================
class MultiLatentAttention(nn.Module):
    def __init__(self, cfg: EvoLLMConfig):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.num_attention_heads = cfg.num_attention_heads
        self.head_dim = cfg.head_dim
        self.q_lora_rank = cfg.q_lora_rank
        self.kv_lora_rank = cfg.kv_lora_rank
        self.is_causal = True

        self.down_q_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=cfg.use_bias)
        self.down_kv_proj = nn.Linear(self.hidden_size, self.kv_lora_rank, bias=cfg.use_bias)
        self.q_norm = RMSNorm(self.q_lora_rank, cfg.rms_norm_eps)
        self.kv_norm = RMSNorm(self.kv_lora_rank, cfg.rms_norm_eps)
        self.up_q_proj = nn.Linear(self.q_lora_rank, self.head_dim * self.num_attention_heads, bias=cfg.use_bias)
        self.up_kv_proj = nn.Linear(self.kv_lora_rank, 2 * self.head_dim * self.num_attention_heads, bias=cfg.use_bias)

        self.o_proj = nn.Linear(self.head_dim * self.num_attention_heads, self.hidden_size, bias=cfg.use_bias)

        self.attn_dropout = nn.Dropout(cfg.dropout_ratio)
        self.resid_dropout = nn.Dropout(cfg.dropout_ratio)

    def forward(self, hidden_states, past_key_value = None, use_cache = False):
        batch_size, seq_len, _ = hidden_states.shape

        # (batch_size, seq_len, kv_lora_rank or q_lora_rank)
        c_kv, c_q = self.down_kv_proj(hidden_states), self.down_q_proj(hidden_states)
        c_kv, c_q = self.kv_norm(c_kv), self.q_norm(c_q)

        if past_key_value is not None:
            c_kv = torch.cat([past_key_value, c_kv], dim=1)
        present_key_value = c_kv if use_cache else None
        kv_len = c_kv.size(1)

        # (batch_size, num_attention_heads, kv_len, head_dim * 2)
        kv = self.up_kv_proj(c_kv).view(batch_size, kv_len, self.num_attention_heads, self.head_dim * 2).transpose(1, 2)
        # 将kv产分为独立的k和v
        k, v = torch.split(kv, [self.head_dim, self.head_dim], dim=-1)
        # (batch_size, num_attention_heads, seq_len, head_dim)
        q = self.up_q_proj(c_q).view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        if self.is_causal:
            causal_mask = torch.full((seq_len, seq_len), float("-inf"), device=hidden_states.device).triu(diagonal=1)
            scores[:, :, :, -seq_len:] += causal_mask

        output = self.attn_dropout(F.softmax(scores, dim=-1)) @ v
        # (batch_size, seq_len, num_attention_heads * head_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        # (batch_size, seq_len, hidden_size)
        output = self.o_proj(output)
        output = self.resid_dropout(output)

        return output, present_key_value


