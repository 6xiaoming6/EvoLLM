import torch

def repeat_kv(x: torch.Tensor, repeat_num: int):
    if repeat_num == 1:
        return x
    batch_size, num_kv_head, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :]
    x = x.expand(batch_size, num_kv_head, repeat_num, seq_len, head_dim)

    # (batch_size, num_attention_heads, seq_len, head_dim)
    return x.reshape(batch_size, num_kv_head * repeat_num, seq_len, head_dim)