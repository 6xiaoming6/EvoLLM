import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import repeat_kv, apply_rope
from models.modules.rms_norm import RMSNorm
from configs.evollm_config import EvoLLMConfig



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

        self.q_lora_rank = cfg.q_lora_rank
        self.kv_lora_rank = cfg.kv_lora_rank
        self.qk_nope_dim = cfg.qk_nope_dim
        self.qk_rope_dim = cfg.qk_rope_dim
        self.q_head_dim = self.qk_rope_dim + self.qk_nope_dim
        self.v_head_dim = cfg.v_head_dim
        self.is_causal = True

        self.down_q_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=cfg.use_bias)
        self.down_kv_proj = nn.Linear(self.hidden_size, self.kv_lora_rank + self.qk_rope_dim, bias=cfg.use_bias)
        self.q_norm = RMSNorm(self.q_lora_rank, cfg.rms_norm_eps)
        self.kv_norm = RMSNorm(self.kv_lora_rank, cfg.rms_norm_eps)
        self.up_q_proj = nn.Linear(self.q_lora_rank, self.num_attention_heads * self.q_head_dim, bias=cfg.use_bias)
        self.up_kv_proj = nn.Linear(self.kv_lora_rank, self.num_attention_heads * (self.qk_nope_dim + self.v_head_dim), bias=cfg.use_bias)

        self.o_proj = nn.Linear(self.v_head_dim * self.num_attention_heads, self.hidden_size, bias=cfg.use_bias)

        self.attn_dropout = nn.Dropout(cfg.dropout_ratio)
        self.resid_dropout = nn.Dropout(cfg.dropout_ratio)

    def forward(self, hidden_states, position_embeddings, past_key_value = None, use_cache = False):
        batch_size, seq_len, _ = hidden_states.shape

        c_q = self.down_q_proj(hidden_states)
        c_q = self.q_norm(c_q)
        # (batch_size, seq_len, num_attention_heads * q_head_dim)
        q = self.up_q_proj(c_q)
        # (batch_size, num_attention_heads, seq_len, q_head_dim)
        q = q.view(batch_size, seq_len, self.num_attention_heads, -1).transpose(1, 2)
        # 把q的nope和rope两部分在最后一个维度拆开
        # (batch_size, num_attention_heads, seq_len, qk_nope_dim or qk_rope_dim)
        q_nope, q_rope = torch.split(q, [self.qk_nope_dim, self.qk_rope_dim], dim=-1)

        c_kv = self.down_kv_proj(hidden_states)
        # (batch_size, seq_len,  kv_low_rank or qk_rope_dim)
        c_kv, k_rope = torch.split(c_kv, [self.kv_lora_rank, self.qk_rope_dim], dim=-1)
        c_kv = self.kv_norm(c_kv)
        # (batch_size, 1, seq_len, qk_rope_dim)
        k_rope = k_rope.unsqueeze(1)

        freqs_sin, freqs_cos = position_embeddings
        q_rope = apply_rope(q_rope, freqs_sin, freqs_cos)
        k_rope = apply_rope(k_rope, freqs_sin, freqs_cos)

        if past_key_value is not None:
            past_c_kv, past_k_rope = past_key_value
            c_kv = torch.cat([past_c_kv, c_kv], dim=1)
            k_rope = torch.cat([past_k_rope, k_rope], dim=2)
        present_key_value = (c_kv, k_rope) if use_cache else None
        kv_len = c_kv.size(1)

        # (batch_size, num_attention_heads, kv_len, qk_nope_dim + v_head_dim)
        kv = self.up_kv_proj(c_kv).view(batch_size, kv_len, self.num_attention_heads, -1).transpose(1, 2)
        # 将kv产分为独立的k_nope和v
        k_nope, v = torch.split(kv, [self.qk_nope_dim, self.v_head_dim], dim=-1)


        nope_scores = q_nope @ k_nope.transpose(-1, -2)
        rope_scores = q_rope @ k_rope.transpose(-1, -2)
        scores = (nope_scores + rope_scores) / math.sqrt(self.q_head_dim)

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

