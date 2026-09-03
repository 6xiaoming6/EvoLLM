import torch
from dataclasses import dataclass
from typing import Literal

@dataclass
class EvoLLMConfig:
    # transformer架构的一些配置
    hidden_size: int = 512
    num_attention_heads: int = 8
    num_kv_heads: int = 8 # 区分mha、mqa和gqa的关键
    num_hidden_layers: int = 8
    intermediate_size: int = 2048
    use_bias: bool = False
    dropout_ratio: float = 0.0
    rms_norm_eps: float = 1e-6
    # mla配置
    q_lora_rank: int = 128
    kv_lora_rank: int = 64
    qk_nope_dim: int = 64
    qk_rope_dim: int = 32
    v_head_dim: int = 64
    # 词表大小
    vocab_size: int = 6400
    bos_token_id: int = 1
    eos_token_id: int = 2
    max_position_embeddings: int = 4096
    rope_base: float = 10000.0

    tie_word_embeddings: bool = True

    @property
    def head_dim(self):
        assert self.hidden_size % self.num_attention_heads == 0
        return self.hidden_size // self.num_attention_heads
    
