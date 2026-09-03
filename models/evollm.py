import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.rms_norm import RMSNorm
from configs.evollm_config import EvoLLMConfig
from models.modules.feed_forward import FeedForward
from models.modules.multi_head_attention import MultiHeadAttention
from utils.utils import precompute_rope_freqs


class EvoLLMBlock(nn.Module):
    def __init__(self, cfg: EvoLLMConfig):
        super().__init__()
        self.input_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attention = MultiHeadAttention(cfg)
        
        self.post_attention_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.ffn = FeedForward(cfg)

    def forward(self, hidden_states, position_embeddings = None, past_key_value = None, use_cache = False):
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)
        hidden_states, present_key_value = self.attention(hidden_states, position_embeddings, past_key_value, use_cache)
        hidden_states += residual

        residual = hidden_states
        hidden_states = self.post_attention_norm(hidden_states)
        output = self.ffn(hidden_states) + residual

        return output, present_key_value


class EvoLLMModel(nn.Module):
    def __init__(self, cfg: EvoLLMConfig):
        super().__init__()
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([
            EvoLLMBlock(cfg) for i in range(cfg.num_hidden_layers)
        ])
        self.dropout = nn.Dropout(cfg.dropout_ratio)
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

        self.cfg = cfg

        sin, cos = precompute_rope_freqs(cfg.max_position_embeddings, cfg.rope_base, cfg.head_dim)
        self.register_buffer("freqs_sin", sin)
        self.register_buffer("freqs_cos", cos)

    def forward(self, input_token_ids, past_key_values = None, use_cache = False):
        batch_size, seq_len = input_token_ids.shape

        # gqa存的形状是 (batch_size, num_heads, seq_len, head_dim)
        start_pos = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        past_key_values = past_key_values or [None] * len(self.layers)

        x = self.dropout(self.embedding(input_token_ids))

        position_embeddings = (self.freqs_sin[start_pos: start_pos + seq_len], self.freqs_cos[start_pos: start_pos + seq_len])

        present_key_values = [] if use_cache else None
        for layer, past_key_value in zip(self.layers, past_key_values):
            x, present_key_value = layer(x, position_embeddings, past_key_value, use_cache)
            if use_cache:
                present_key_values.append(present_key_value)
        
        output = self.norm(x)

        return output, present_key_values


class EvoLLMForCausalLM(nn.Module):
    def __init__(self, cfg: EvoLLMConfig):
        super().__init__()
        self.vocab_size = cfg.vocab_size
        self.hidden_size = cfg.hidden_size

        self.model = EvoLLMModel(cfg)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=cfg.use_bias)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embedding.embedding.weight

        # 递归的给各层初始化参数权重
        self.apply(self._init_weights)


    # 使用均值为 0、标准差为 0.02 的正态分布初始化权重
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_token_ids, labels = None, past_key_values = None, use_cache = False):
        batch_size, seq_len = input_token_ids.shape

        #(bsz, seq_len, hidden_size)
        hidden_states, present_key_values = self.model(input_token_ids, past_key_values, use_cache)
        #(bsz, seq_len, vocab_size)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., : -1, :].contiguous()
            shift_labels = labels[..., 1: ].contiguous()

            # 每条序列最后的padding会被标记为-100，不参计算该位置的计算
            loss = F.cross_entropy(shift_logits.view(-1, self.vocab_size), 
                                   shift_labels.view(-1), 
                                   ignore_index=-100
                                   )

        return{
            "logits": logits,
            "loss": loss,
            "hidden_states": hidden_states,
            "present_key_values": present_key_values
        }