import torch

mask = torch.full((1, 1), float("-inf")).triu(diagonal=1)
print(mask)