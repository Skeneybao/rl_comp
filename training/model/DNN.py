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
    def __init__(self, config: DNNModelConfig):
        super().__init__()
        self.fc_in = nn.Linear(config.input_dim, config.hidden_dim[0])
        self.fcs = [nn.Linear(config.hidden_dim[i], config.hidden_dim[i + 1])
                    for i in range(len(config.hidden_dim) - 1)]
        self.fc_out = nn.Linear(config.hidden_dim[-1], config.output_dim)

        self.activation = activations[config.activation]
        if config.output_activation is not None:
            self.output_activation = activations[config.output_activation]
        else:
            self.output_activation = None

    def forward(self, x):
        x = self.activation(self.fc_in(x))
        for fc in self.fcs:
            x = self.activation(fc(x))
        x = self.fc_out(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x
