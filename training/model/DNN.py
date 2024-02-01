from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn

from training.model.activations import activations


@dataclass
class DNNModelConfig:
    input_dim: int
    hidden_dim: List[int]
    output_dim: int
    activation: str = 'gelu'
    output_activation: Optional[str] = None


class DNN(nn.Module):
    def __init__(
            self,
            *,
            input_dim: int,
            hidden_dim: List[int],
            output_dim: int,
            activation: str = 'gelu',
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


class DNN_11_output(nn.Module):
    def __init__(
            self,
            *,
            input_dim: int,
            hidden_dim: List[int],
            output_dim: int,
            activation: str = 'gelu',
            output_activation: Optional[str] = None,
            **kwargs
    ):
        super().__init__()
        output_dim = 1

        self.fc_in = nn.Linear(input_dim, hidden_dim[0])
        self.hidden = nn.ModuleList()
        for i in range(len(hidden_dim) - 2):
            self.hidden.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
        self.hidden.append(nn.Linear(hidden_dim[-2] + 1, hidden_dim[-1]))
        self.fc_out = nn.Linear(hidden_dim[-1], output_dim)

        self.activation = activations[activation]
        if output_activation is not None:
            self.output_activation = activations[output_activation]
        else:
            self.output_activation = None

    def forward(self, x):
        # VERY IMPORTANT:
        # assuming
        # 1. the last 10 features are the top 10 features, with order in -5, -4, -3, -2, -1, 1, 2, 3, 4, 5
        #     i.e., with order of sell 5, sell 4, sell 3, sell 2, sell 1, buy 1, buy 2, buy 3, buy 4, buy 5
        # 2. the id in output wrapper is in the order of
        #     buy 5, buy 4, buy 3, buy 2, buy 1, sell 1, sell 2, sell 3, sell 4, sell 5
        no_batch = False
        if len(x.size()) == 1:
            no_batch = True
            x = x.unsqueeze(0)

        op_features = x[:, -10:]  # bs, 10
        # reverse the order of op_features
        op_features = torch.flip(op_features, dims=[-1])  # bs, 10
        op_features_with_noop = torch.cat([op_features, torch.zeros_like(op_features[:, :1])], dim=-1)  # bs, 11
        x = self.activation(self.fc_in(x))
        for layer in self.hidden[:-1]:
            x = self.activation(layer(x))

        # duplicate x for 11 times
        x = x.unsqueeze(1).repeat(1, 11, 1)  # bs, 11, hidden_dim[-2]
        # concatenate op_features to x
        x = torch.cat([x, op_features_with_noop.unsqueeze(2)], dim=-1)  # bs, 11, hidden_dim[-2] + 1
        x = self.activation(self.hidden[-1](x))

        x = self.fc_out(x).squeeze(-1)
        if self.output_activation is not None:
            x = self.output_activation(x)

        if no_batch:
            x = x.squeeze(0)
        return x


class FullPosDNN(DNN):
    def __init__(
            self,
            *,
            input_dim: int,
            hidden_dim: List[int],
            output_dim: int,
            activation: str = 'gelu',
            output_activation: Optional[str] = None,
            full_signal_pos: int = 0,
    ):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                         activation=activation, output_activation=output_activation)
        self.full_signal_pos = full_signal_pos

    def forward(self, x):
        original_forward = super().forward(x)

        is_batched = len(original_forward.shape) > 1
        if not is_batched:
            x = x.unsqueeze(0)
            original_forward = original_forward.unsqueeze(0)

        full_signal = x[:, self.full_signal_pos]

        res = torch.zeros_like(original_forward)
        res[:, 0] = torch.where(full_signal > 0, torch.tensor(-torch.inf), original_forward[:, 0])
        res[:, 2] = torch.where(full_signal < 0, torch.tensor(-torch.inf), original_forward[:, 2])
        res[:, 1] = original_forward[:, 1]

        if not is_batched:
            res = res.squeeze(0)

        return res
