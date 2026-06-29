import torch

from input_embedding import InputEmbedding
from decoder_layer import DecoderLayer


class Decoder(torch.nn.Module):

    def __init__(
            self,
            dec_voc_size: int,
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
            vocab_size=dec_voc_size,
            dropout_prob=dropout_prob,
            device=device
        )
        self.layers = torch.nn.ModuleList([
            DecoderLayer(
                d_model=d_model,
                ffn_hidden=ffn_hidden,
                num_heads=num_heads,
                dropout_prob=dropout_prob
            )
            for _ in range(n_layers)
        ])
        self.linear = torch.nn.Linear(d_model, dec_voc_size)

    def forward(
            self,
            trg: torch.Tensor,
            enc_src: torch.Tensor,
            trg_mask=None,
            src_mask=None,
    ) -> torch.Tensor:
        trg = self.emb(trg)
        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)
        output = self.linear(trg)
        return output
