import torch

from feed_forward import FeedForward
from multi_head_attention import MultiHeadAttention
from residual_connection import ResidualConnection


class EncoderLayer(torch.nn.Module):

    def __init__(
            self,
            d_model: int,
            ffn_hidden: int,
            num_heads: int,
            drop_prob: float,
    ) -> None:
        super().__init__()

        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.ffn = FeedForward(d_model=d_model, d_ff=ffn_hidden, dropout_prob=drop_prob)
        self.res1 = ResidualConnection(d_model, drop_prob)
        self.res2 = ResidualConnection(d_model, drop_prob)

    def forward(self, x: torch.Tensor, src_mask: bool) -> torch.Tensor:
        attn_out = self.attention(q=x, k=x, v=x, mask=src_mask)
        x = self.res1(x, attn_out)
        ffn_out = self.ffn(x)
        x = self.res2(x, ffn_out)
        return x
