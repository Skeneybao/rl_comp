from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from training.model.activations import activations


@dataclass
class AttnModelConfig:
    input_dim: int
    hidden_dim: List[int]
    avg_price_dim: int
    output_dim: int
    activation: str = 'gelu'
    output_activation: Optional[str] = None


class AttentionLayer(nn.Module):
    def __init__(self, feature_dim, avg_price_dim):
        super(AttentionLayer, self).__init__()
        self.query = nn.Linear(feature_dim, avg_price_dim)
        self.key = nn.Linear(10, avg_price_dim)
        self.value = nn.Linear(10, avg_price_dim)

    def forward(self, features, avg_prices):
        query = self.query(features)
        key = self.key(avg_prices)
        value = self.value(avg_prices)

        # Scaled Dot-Product Attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / (avg_prices.size(-1) ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, value)

        return output


class Attn(nn.Module):
    def __init__(
            self,
            *,
            input_dim: int,
            avg_price_dim: int = 16,  # Dimension of avgPrice features
            hidden_dim: List[int],
            output_dim: int,
            activation: str = 'gelu',
            output_activation: Optional[str] = None
    ):
        super().__init__()
        self.fc_in = nn.Linear(input_dim - 10, hidden_dim[0])
        self.hidden = nn.ModuleList()
        for i in range(len(hidden_dim) - 1):
            self.hidden.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))

        self.attention = AttentionLayer(hidden_dim[-1], avg_price_dim)

        self.fc_out = nn.Linear(hidden_dim[-1] + avg_price_dim, output_dim)

        self.activation = activations[activation]
        if output_activation is not None:
            self.output_activation = activations[output_activation]
        else:
            self.output_activation = None

    def forward(self, x):
        no_batch = False
        if len(x.size()) == 1:
            no_batch = True
            x = x.unsqueeze(0)

        avg_prices = x[:, -10:]  # Last 10 dimensions are avgPrice
        other_features = x[:, :-10]  # Rest of the features

        x = self.activation(self.fc_in(other_features))
        for layer in self.hidden:
            x = self.activation(layer(x))

        attention_output = self.attention(x, avg_prices)
        x = torch.cat([x, attention_output], dim=-1)

        x = self.fc_out(x)
        if self.output_activation is not None:
            x = self.output_activation(x)

        if no_batch:
            x = x.squeeze(0)
        return x
