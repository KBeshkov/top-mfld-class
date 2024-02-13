import torch
import torch.nn as nn

class FeedforwardNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, activation=nn.ReLU(), out_layer_sz = 0, mean=-0.5, std=2, init_type = 'uniform'):
        super(FeedforwardNetwork, self).__init__()

        self.layers = nn.ModuleList()
        
        self.mean = mean
        self.std = std
        
        self.init_type = init_type
        self.out_layer_sz = out_layer_sz

        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(activation)

        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.layers.append(activation)
        if out_layer_sz>0:
            self.layers.append(nn.Linear(hidden_sizes[-1],out_layer_sz))

        # Initialize weights
        self.initialize_weights(mean, std)

    def forward(self, x):
        layerwise_activations = []
        for layer in self.layers:
            if isinstance(layer, nn.ReLU):
                x = layer(x)
                layerwise_activations.append(x.clone())
            else:
                x = layer(x)
        if self.out_layer_sz>0:
            layerwise_activations.append(x)
        return layerwise_activations

    def initialize_weights(self, mean, std):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                if self.init_type == 'normal':
                    nn.init.normal_(layer.weight, mean=mean, std=std)
                elif self.init_type == 'uniform':
                    std*(nn.init.uniform_(layer.weight,0,1)+mean)
                elif self.init_type == 'zeros':
                    nn.init.zeros_(layer.weight)
                    # nn.init.normal_(layer.bias,mean=0, std=std)
                # nn.init.zeros_(layer.bias)#,-std,std)

