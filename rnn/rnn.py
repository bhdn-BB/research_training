import torch


class RNN(torch.nn.Module):
    def __init__(
            self,
            x_size: int,
            h_size: int,
            output_size: int,
            num_layers: int,
    ) -> None:
        super().__init__()

        self.x_size = x_size
        self.h_size = h_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.wx = torch.nn.ModuleList()
        self.wh = torch.nn.ModuleList()

        for i in range(num_layers):
            in_size = x_size if i == 0 else h_size
            self.wx.append(torch.nn.Linear(in_size, h_size))
            self.wh.append(torch.nn.Linear(h_size, h_size))

        self.wy = torch.nn.Linear(h_size, output_size)

    def forward(self, input_sequences: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = input_sequences.shape
        h = [
            torch.zeros(batch_size, self.h_size, device=input_sequences.device)
            for _ in range(self.num_layers)
        ]
        outputs = torch.empty(
            batch_size,
            seq_len,
            self.output_size,
            device=input_sequences.device
        )
        for t in range(seq_len):
            x_t = input_sequences[:, t, :]
            for l in range(self.num_layers):
                inp = x_t if l == 0 else h[l - 1]
                h[l] = torch.tanh(self.wx[l](inp) + self.wh[l](h[l]))
            outputs[:, t, :] = self.wy(h[-1])
        return outputs