import torch

from input_embedding import InputEmbedding
from encoder_layer import EncoderLayer


class Encoder(torch.nn.Module):

    def __init__(
        self,
        enc_voc_size: int,
        max_len: int,
        d_model: int,
        ffn_hidden: int,
        num_heads: int,
        n_layers: int,
        dropout_prob: float,
        device
    ) -> None:
        super().__init__()

        self.emb = InputEmbedding(
            d_model=d_model,
            max_len=max_len,
            vocab_size=enc_voc_size,
            dropout_prob=dropout_prob,
            device=device
        )

        self.layers = torch.nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                ffn_hidden=ffn_hidden,
                num_heads=num_heads,
                drop_prob=dropout_prob
            )
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, src_mask=None) -> torch.Tensor:
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x
