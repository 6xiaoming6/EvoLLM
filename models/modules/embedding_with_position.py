import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.evollm_config import EvoLLMConfig

def precompute_position_embeddings(n, dim):
    pos_embeding = torch.zeros((n, dim))
    for pos in range(n):
        for i in range(dim // 2):
            pos_embeding[pos][2 * i] = math.sin(pos / math.pow(10000, 2 * i / dim))
            pos_embeding[pos][2 * i + 1] = math.cos(pos / math.pow(10000, 2 * i / dim))
    return pos_embeding

class EmbeddingWithPosition(nn.Module):
    def __init__(self, cfg: EvoLLMConfig):
        super().__init__()
        self.dim = cfg.hidden_size
        self.embedding = nn.Embedding(cfg.vocab_size, self.dim)
        pos_embed = precompute_position_embeddings(cfg.max_position_embeddings, self.dim)
        self.register_buffer("pos_embed", pos_embed)

    def forward(self, input_token_ids):
        batch_size, seq_len = input_token_ids.shape
        return self.embedding(input_token_ids) + self.pos_embed[:seq_len, :]

