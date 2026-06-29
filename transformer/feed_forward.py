import torch


class FeedForward(torch.nn.Sequential):
    def __init__(self, d_model: int, d_ff: int, dropout_prob: float) -> None:
        super(FeedForward).__init__(
            torch.nn.Linear(d_model, d_ff),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_prob),
            torch.nn.Linear(d_ff, d_model)
        )
