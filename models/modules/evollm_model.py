import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.rms_norm import RMSNorm
from configs.evollm_config import EvoLLMConfig
from models.modules.evollm_block import EvoLLMBlock
from models.modules.embedding_with_position import EmbeddingWithPosition

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
        
