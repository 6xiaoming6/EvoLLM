import torch

mask = torch.full((5, 5), float("-inf")).triu(diagonal=1)
print(mask)
print(1 / 2)