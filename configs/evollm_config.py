from dataclasses import dataclass
import torch

@dataclass
class EvoLLMConfig:
    # transformer架构的一些配置
    hidden_size: int = 512
    num_attention_heads: int = 8
    num_hidden_layers: int = 8
    intermediate_size: int = 2048
    use_bias: bool = False
    dropout_ratio: float = 0.0
    rms_norm_eps: float = 1e-6

    max_position_embeddings: int = 2048
    # 词表大小
    vocab_size: int = 6400
    bos_token_id: int = 1
    eos_token_id: int = 2

    tie_word_embeddings: bool = True

    @property
    def head_dim(self):
        assert self.hidden_size % self.num_heads == 0
        return self.hidden_size // self.num_heads
    
