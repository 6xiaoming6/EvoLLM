import torch

def repeat_kv(x: torch.Tensor, repeat_num: int):
    if repeat_num == 1:
        return x
    batch_size, num_kv_head, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :]
    x = x.expand(batch_size, num_kv_head, repeat_num, seq_len, head_dim)

    # (batch_size, num_attention_heads, seq_len, head_dim)
    return x.reshape(batch_size, num_kv_head * repeat_num, seq_len, head_dim)


def precompute_rope_freqs(max_position_embeddings, base, dim):
    position_idx = torch.arange(0, max_position_embeddings)
    assert dim % 2 == 0
    thetas = torch.pow(base, -torch.arange(0, dim, 2) / dim)
    thetas = torch.cat([thetas, thetas])
    freqs = torch.outer(position_idx, thetas)

    return torch.sin(freqs), torch.cos(freqs)

def rotate_half(x):
    x_pre = x[..., : x.shape[-1] // 2]
    x_post = x[..., x.shape[-1] // 2: ]

    return torch.cat([-x_post, x_pre], dim=-1)

def apply_rope(x, sin, cos):
    return x * cos + rotate_half(x) * sin