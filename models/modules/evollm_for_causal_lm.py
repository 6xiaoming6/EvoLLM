import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.evollm_config import EvoLLMConfig
from models.modules.evollm_model import EvoLLMModel


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