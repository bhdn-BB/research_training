import torch


class SelfAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            mask = None,
    ) -> torch.Tensor:
        batch_size, head, length, d_tensor = k.size()
        # [batch, heads, seq_len, d_k] -> [batch, heads, d_k, seq_len]
        k_t = k.transpose(2, 3)
        attention_matrix = (q @ k_t) / d_tensor ** 0.5
        if mask is not None:
            attention_matrix = attention_matrix.masked_fill(mask == 0, -1e6)
        attention_matrix = torch.softmax(attention_matrix, dim=-1)
        v = attention_matrix @ v
        return v
