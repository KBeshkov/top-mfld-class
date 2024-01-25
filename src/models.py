import torch
import torch.nn as nn

class FeedforwardNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation=nn.ReLU(), mean=0, std=0.01):
        super(FeedforwardNetwork, self).__init__()

        self.layers = nn.ModuleList()
        
        self.mean = mean
        self.std = std

        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(activation)

        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.layers.append(activation)

        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

        # Initialize weights
        self.initialize_weights(mean, std)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def initialize_weights(self, mean, std):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=mean, std=std)
