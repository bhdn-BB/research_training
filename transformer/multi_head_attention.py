import torch

from self_attention import SelfAttention


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.attention = SelfAttention()

        self.w_q = torch.nn.Linear(d_model, d_model)
        self.w_k = torch.nn.Linear(d_model, d_model)
        self.w_v = torch.nn.Linear(d_model, d_model)
        self.w_o = torch.nn.Linear(d_model, d_model)

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            mask=None,
    ) -> torch.Tensor:

        q = self.w_q(q)
        q = self._split(q)

        k = self.w_k(k)
        k = self._split(k)

        v = self.w_v(v)
        v = self._split(v)

        features = self.attention(q, k, v, mask=mask)
        features = self.w_0(self._concat_tensors(features))
        return features

    def _split(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, length, d_model = tensor.size()
        d_hidden = d_model // self.num_heads
        tensor = (tensor
                  .view(batch_size, length, self.num_heads, d_hidden)
                  .transpose(1, 2))
        return tensor

    def _concat_tensors(self, tensor):
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor
        tensor = (tensor
                  .transpose(1, 2)
                  .contiguous()
                  .view(batch_size, length, d_model))
        return tensor
