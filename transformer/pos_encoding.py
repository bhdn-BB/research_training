import torch


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, max_len: int, device: torch.device) -> None:
        super().__init__()
        encoding = torch.zeros(max_len, d_model, device=device)
        encoding.requires_grad = False
        pos = torch.arange(max_len, device=device).float().view(-1, 1)
        i = torch.arange(0, d_model, step=2, device=device).float()
        encoding[:, 0::2] = torch.sin(pos / 10000 ** (i / d_model))
        encoding[:, 1::2] = torch.cos(pos / 10000 ** (i / d_model))
        self.register_buffer("encoding", encoding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return self.encoding[:seq_len]
