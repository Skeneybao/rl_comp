from dataclasses import dataclass
from typing import List, Optional

from torch import nn

from training.model.activations import activations


@dataclass
class DNNModelConfig:
    input_dim: int
    hidden_dim: List[int]
    output_dim: int
    activation: str = 'relu'
    output_activation: Optional[str] = None


class DNN(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: List[int],
            output_dim: int,
            activation: str = 'relu',
            output_activation: Optional[str] = None
    ):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim[0])
        self.hidden = nn.ModuleList()
        for i in range(len(hidden_dim) - 1):
            self.hidden.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
        self.fc_out = nn.Linear(hidden_dim[-1], output_dim)

        self.activation = activations[activation]
        if output_activation is not None:
            self.output_activation = activations[output_activation]
        else:
            self.output_activation = None

    def forward(self, x):
        x = self.activation(self.fc_in(x))
        for layer in self.hidden:
            x = self.activation(layer(x))
        x = self.fc_out(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x
