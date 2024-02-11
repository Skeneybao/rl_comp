from typing import List, Optional

from torch import nn

from training.model.activations import activations


class QRDNN(nn.Module):
    def __init__(
            self,
            *,
            input_dim: int,
            hidden_dim: List[int],
            output_dim: int,
            activation: str = 'gelu',
            output_activation: Optional[str] = None,
            quant_dim: int = 16
    ):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim[0])
        self.hidden = nn.ModuleList()
        for i in range(len(hidden_dim) - 1):
            self.hidden.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
        self.fc_out = nn.Linear(hidden_dim[-1], output_dim * quant_dim)

        self.output_dim = output_dim
        self.quant_dim = quant_dim

        self.activation = activations[activation]
        if output_activation is not None:
            self.output_activation = activations[output_activation]
        else:
            self.output_activation = None

    def forward(self, x):
        is_batched = len(x.shape) > 1
        if not is_batched:
            x = x.unsqueeze(0)
        x = self.activation(self.fc_in(x))
        for layer in self.hidden:
            x = self.activation(layer(x))
        x = self.fc_out(x)
        x = x.view(x.shape[0], -1, self.quant_dim)
        if self.output_activation is not None:
            x = self.output_activation(x, dim=2)

        if not is_batched:
            x = x.squeeze(0)
        return x
