from torch import nn

activations = {
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
    'softmax': nn.Softmax(),
    'gelu': nn.GELU(approximate='tanh'),
    'leaky_relu': nn.LeakyReLU(),
    'elu': nn.ELU(),
}
