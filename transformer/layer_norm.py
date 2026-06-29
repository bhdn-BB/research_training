import torch


class LayerNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = torch.nn.Parameter(torch.ones(d_model))
        self.bias = torch.nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_mean = x.mean(dim=-1, keepdim=True)
        x_std = x.std(dim=-1, keepdim=True)
        x_norm = (x - x_mean) / (x_std + self.eps)
        return self.alpha * x_norm + self.bias
