import torch


class GRU(torch.nn.Module):
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

        self.wz_x = torch.nn.ModuleList()
        self.wz_h = torch.nn.ModuleList()

        self.wr_x = torch.nn.ModuleList()
        self.wr_h = torch.nn.ModuleList()

        self.wn_x = torch.nn.ModuleList()
        self.wn_h = torch.nn.ModuleList()

        for i in range(num_layers):
            in_size = x_size if i == 0 else h_size

            self.wz_x.append(torch.nn.Linear(in_size, h_size))
            self.wz_h.append(torch.nn.Linear(h_size, h_size))

            self.wr_x.append(torch.nn.Linear(in_size, h_size))
            self.wr_h.append(torch.nn.Linear(h_size, h_size))

            self.wn_x.append(torch.nn.Linear(in_size, h_size))
            self.wn_h.append(torch.nn.Linear(h_size, h_size))

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
                z = torch.sigmoid(self.wz_x[l](inp) + self.wz_h[l](h[l]))
                r = torch.sigmoid(self.wr_x[l](inp) + self.wr_h[l](h[l]))
                h_candidate = torch.tanh(self.wn_x[l](inp) + self.wn_h[l](r * h[l]))
                h[l] = z * h[l] + (1 - z) * h_candidate
            outputs[:, t, :] = self.wy(h[-1])
        return outputs
