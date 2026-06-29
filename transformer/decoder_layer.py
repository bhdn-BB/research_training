import torch
import torch.nn as nn

from multi_head_attention import MultiHeadAttention
from feed_forward import FeedForward
from residual_connection import ResidualConnection


class DecoderLayer(nn.Module):

    def __init__(
            self,
            d_model: int,
            ffn_hidden: int,
            num_heads: int,
            dropout_prob: float
    ) -> None:
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.cross_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

        self.ffn = FeedForward(d_model=d_model, d_ff=ffn_hidden, dropout_prob=dropout_prob)

        self.res1 = ResidualConnection(d_model, dropout_prob)
        self.res2 = ResidualConnection(d_model, dropout_prob)
        self.res3 = ResidualConnection(d_model, dropout_prob)

    def forward(
            self,
            dec: torch.Tensor,
            enc: torch.Tensor,
            trg_mask=None,
            src_mask=None,
    ) -> torch.Tensor:
        self_attn_out = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)
        dec = self.res1(dec, self_attn_out)
        if enc is not None:
            cross_attn_out = self.cross_attention(q=dec, k=enc, v=enc, mask=src_mask)
            dec = self.res2(dec, cross_attn_out)
        ffn_out = self.ffn(dec)
        dec = self.res3(dec, ffn_out)
        return dec
