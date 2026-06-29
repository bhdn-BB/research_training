import torch

from transformer.layer_norm import LayerNorm


class ResidualConnection(torch.nn.Module):
    def __init__(self, features: int, dropout_prob: float) -> None:
        super().__init__()
        self.norm = LayerNorm(features)
        self.dropout = torch.nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor, sublayer: torch.nn.Module) -> torch.Tensor:
        return x + self.dropout(sublayer(self.norm(x)))
