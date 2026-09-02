import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.evollm_config import EvoLLMConfig

class Embedding(nn.Module):
    def __init__(self, cfg: EvoLLMConfig):
        super().__init__()
        self.dim = cfg.hidden_size
        self.embedding = nn.Embedding(cfg.vocab_size, self.dim)

    def forward(self, input_token_ids):
        return self.embedding(input_token_ids)

