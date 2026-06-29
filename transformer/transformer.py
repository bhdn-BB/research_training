import torch

from encoder import Encoder
from decoder import Decoder


class Transformer(torch.nn.Module):

    def __init__(
            self,
            src_pad_idx: int,
            trg_pad_idx: int,
            trg_sos_idx: int,
            enc_voc_size: int,
            dec_voc_size: int,
            d_model: int,
            num_heads: int,
            max_len: int,
            ffn_hidden: int,
            n_layers: int,
            dropout_prob: float,
            device
    ) -> None:
        super().__init__()

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device

        self.encoder = Encoder(
            enc_voc_size=enc_voc_size,
            max_len=max_len,
            d_model=d_model,
            ffn_hidden=ffn_hidden,
            num_heads=num_heads,
            n_layers=n_layers,
            dropout_prob=dropout_prob,
            device=device
        )
        self.decoder = Decoder(
            dec_voc_size=dec_voc_size,
            max_len=max_len,
            d_model=d_model,
            ffn_hidden=ffn_hidden,
            num_heads=num_heads,
            n_layers=n_layers,
            dropout_prob=dropout_prob,
            device=device
        )

    def forward(self, src: torch.Tensor, trg: torch.Tensor):
        src_mask = self._make_src_mask(src)
        trg_mask = self._make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output

    def _make_src_mask(self, src: torch.Tensor):
        # 1 = real token, 0 = padding
        mask = (src != self.src_pad_idx)
        # [batch, 1, 1, seq_len]
        return mask.unsqueeze(1).unsqueeze(2)

    def _make_trg_mask(self, trg: torch.Tensor):
        # padding mask
        trg_pad_mask = (trg != self.trg_pad_idx)
        trg_pad_mask = trg_pad_mask.unsqueeze(1).unsqueeze(3)
        # causal mask
        trg_len = trg.shape[1]
        causal_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        # combine masks
        trg_mask = trg_pad_mask & causal_mask
        return trg_mask
