import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.rms_norm import RMSNorm
from configs.evollm_config import EvoLLMConfig
from models.modules.attention import Attention
from models.modules.feed_forward import FeedForward
from models.modules.embedding_with_position import EmbeddingWithPosition

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


class EvoLLMModel(nn.Module):
    def __init__(self, cfg: EvoLLMConfig):
        super().__init__()
        self.embedding = EmbeddingWithPosition(cfg)
        self.layers = nn.ModuleList([
            EvoLLMBlock(cfg) for i in range(cfg.num_hidden_layers)
        ])
        self.dropout = nn.Dropout(cfg.dropout_ratio)
        self.norm = RMSNorm(cfg)

    def forward(self, input_token_ids):
        batch_size, seq_len = input_token_ids.shape

        x = self.dropout(self.embedding(input_token_ids))
        for layer in self.layers:
            x = layer(x)
        output = self.norm(x)

        return output


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

    # 使用均值为 0、标准差为 0.02 的截断正态分布初始化权重
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_token_ids, labels = None):
        batch_size, seq_len = input_token_ids.shape

        #(bsz, seq_len, hidden_size)
        hidden_states = self.model(input_token_ids)
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
            "hidden_states": hidden_states
        }