import torch

from pos_encoding import PositionalEncoding


class InputEmbedding(torch.nn.Module):
    def __init__(
            self,
            d_model: int,
            vocab_size: int,
            max_len: int,
            dropout_prob: float,
            device: torch.device,
    ) -> None:
        super().__init__()

        self.tok_emb = torch.nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = torch.nn.Dropout(p=dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)
