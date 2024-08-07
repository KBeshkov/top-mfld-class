import torch
import torch.nn as nn
import sys
sys.path.append('/Users/kosio/Repos/LUP-rank-computer/')
from LUP_rank import rank_revealing_LUP
import numpy as np
from scipy.optimize import curve_fit


class FeedforwardNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, activation=nn.ReLU, out_layer_sz = 0, mean=-0.5, std=2, init_type = 'uniform',conv_dict={}):
        super(FeedforwardNetwork, self).__init__()

        self.layers = nn.ModuleList()
        self.activation = activation
        
        self.mean = mean
        self.std = std
        
        self.init_type = init_type
        self.out_layer_sz = out_layer_sz

        self.conv_dict = conv_dict
        # Input layer
        if len(conv_dict)>0:
            # self.layers_ff = nn.ModuleList()
            
            self.in_channels = conv_dict['in_channels']
            self.out_channels = conv_dict['out_channels']
            self.kernel_sizes = conv_dict['kernel_sizes']
            self.image_size = conv_dict['image_size'] #only valid for square images so far
            
            self.feature_layer = []
            
            self.feature_layer.append(nn.Conv2d(self.in_channels,self.out_channels[0],self.kernel_sizes[0],bias=False))
            self.feature_layer.append(activation())
            self.feature_layer.append(nn.MaxPool2d(2,1))
            
            # self.layers_ff.append(nn.Linear(self.in_channels*self.image_size**2, self.out_channels[0]*(self.image_size-self.kernel_sizes[0]+1)**2,bias=False))
            # self.layers_ff.weight = self.convolution_to_dense(self.layers[0], self.image_size)
            # self.layers_ff.append(activation())
            
            self.grid_size = self.image_size-self.kernel_sizes[0]
            for i in range(1,len(self.out_channels)):
                self.feature_layer.append(nn.Conv2d(self.out_channels[i-1], self.out_channels[i], self.kernel_sizes[i],bias=False))
                self.feature_layer.append(activation())
                self.feature_layer.append(nn.MaxPool2d(2,1))

                # self.layers_ff.append(nn.Linear(self.out_channels[i-1]*self.grid_size**2, self.out_channels[i]*(self.grid_size-self.kernel_sizes[i]+1)**2,bias=False))
                # self.layers_ff.weight = self.convolution_to_dense(self.layers[-2], self.grid_size)
                
                self.grid_size = self.grid_size - self.kernel_sizes[i]
            self.flat_size = self.out_channels[-1]*self.grid_size**2
            self.feature_layer.append(nn.Flatten())
            self.layers.append(nn.Linear(self.flat_size, hidden_sizes[0]))
            self.layers.append(activation())
            
            self.feature_layer = nn.Sequential(*self.feature_layer)
            # self.layers_ff.append(self.layers[-2])
        else:
            self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
            self.layers.append(activation())
            
            # self.layers_ff.append(self.layers[0])
            # self.layers_ff.append(activation())

        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.layers.append(activation())
            
            # self.layers_ff.append(self.layers[-2])
            # self.layers_ff.append(activation())
        if out_layer_sz>0:
            self.layers.append(nn.Linear(hidden_sizes[-1],out_layer_sz))
            
            # self.layers_ff.append(self.layers[-1])

        # Initialize weights
        self.initialize_weights(mean, std)

    def forward(self, x):
        layerwise_activations = []
        if len(self.conv_dict)>0:
            x = self.feature_layer(x)
        for layer in self.layers:
            if isinstance(layer, self.activation):
                x = layer(x)
                layerwise_activations.append(torch.flatten(x.clone(),1))
            else:
                x = layer(x)
        if self.out_layer_sz>0:
            layerwise_activations.append(x)
        return layerwise_activations

    def initialize_weights(self, mean, std):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # nn.init.zeros_(layer.bias)#,-std,std)
                if self.init_type == 'normal':
                    nn.init.normal_(layer.weight, mean=mean, std=std)
                elif self.init_type == 'uniform':
                    std*(nn.init.uniform_(layer.weight,0,1)+mean)
                elif self.init_type == 'zeros':
                    nn.init.zeros_(layer.weight)
                elif self.init_type == 'orthogonal':
                    nn.init.orthogonal_(layer.weight)
                else:
                    break
                    # nn.init.normal_(layer.bias,mean=0, std=std)
                    
    # def kernel_to_dense(self,k,N):
    #     n = len(k)
    #     W = torch.zeros([(N-n+1)**2,N**2])
    #     indexing_sequence = torch.concatenate([torch.arange(k*n,(k+1)*n)+k*(N-n) for k in range(n)])
    #     n_counter = 0
    #     for i in range(len(W)):
    #         if i>0 and i%(N-n+1)==0:
    #             n_counter+=n-1
    #         W[i,indexing_sequence+i+n_counter] = k.flatten()
    #     return torch.Tensor(W)

    # def convolution_to_dense(self,conv,N):
    #     C_in = conv.weight.shape[1]
    #     C_out = conv.weight.shape[0]
    #     kern_size = conv.weight.shape[2]
    #     W = torch.zeros([C_out,(N-kern_size+1)**2, N**2,C_in])
    #     for j in range(C_out):
    #         for i in range(C_in):
    #             W[j,:,:,i] = self.kernel_to_dense(conv.weight[j,i],N)
    #     w = torch.cat([W[:,:,:,i] for i in range(C_in)],-1)
    #     w = torch.cat([w[j] for j in range(C_out)],0)
    #     return w
    
                    
    def compute_rank(self, out_sign):
        out_mat = 0
        ranks = []
        for x in out_sign:
            for layer in self.layers:
                Q = (torch.eye(layer.weight.shape[0])*x)
                out_mat = Q@layer.weight
            try:
                fin_rank = rank_revealing_LUP(out_mat.detach().numpy().astype('float')) #change from torch.matrix_rank for speedup
            except:
                fin_rank = torch.lingalg.matrix_rank(out_mat)
            ranks.append(fin_rank.item())
        return ranks
    
    def compute_rank_at_point(self, x):
        ranks = []
        if len(self.conv_dict)>0:
            x = self.feature_layer(x)
        for n, layer in enumerate(self.layers):
            x = layer(x.flatten())
            if isinstance(layer, nn.Linear):
                out_sign = (x>0).float()
                Q = (torch.eye(layer.weight.shape[0])*out_sign)
                if n==0:
                    out_mat = Q@layer.weight
                else:
                    out_mat = Q@layer.weight@out_mat
                try:
                    fin_rank = rank_revealing_LUP(out_mat.detach().numpy().astype('float'))
                except:
                    fin_rank = torch.linalg.matrix_rank(out_mat)
                ranks.append(fin_rank)
        return ranks
    
    
    # def compute_rank_jacobian(self, x):
    #     ranks = []
    #     jacobian = []
    #     x_grad = torch.clone(x)
    #     x_grad.requires_grad = True
    #     if len(self.conv_dict)>0:
    #         x = self.feature_layer(x)
    #     y = self.forward(x.reshape(1,-1))[-2]
    #     y.requires_grad = True
    #     for i in range(y.size(1)):
    #         grad_output = torch.zeros_like(y)

    #         grad_output[:, i] = 1
    #         gradients = torch.autograd.grad(outputs=y, inputs=x_grad, grad_outputs=grad_output, retain_graph=True, create_graph=True)[0]
    #         jacobian.append(gradients)

    #     jacobian = torch.stack(jacobian, dim=1)
    #     sing_vals = torch.linalg.svd(jacobian)[1]
    #     sing_vals = sing_vals/torch.max(sing_vals) #normalize top singular value to 1
    #     exp_constant = 1/curve_fit(self.exp_model, np.arange(0, len(sing_vals)), sing_vals, p0=0.01)[0]
            
    #     ranks.append(exp_constant)
    #     return ranks

       
def exp_model(x, a):
    return np.exp(-a*x)


def compute_jacobian_rank(model, x):
    ranks = []
    jacobian = []
    x = x.reshape(len(x),-1)
    x.requires_grad = True
    y = model(x)[-2] #get penultimate layer jacobian
    for i in range(y.size(1)):
        grad_output = torch.zeros_like(y)
        grad_output[:, i] = 1
        gradients = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=grad_output, retain_graph=True, create_graph=True)[0]
        jacobian.append(gradients)
    
    jacobian = torch.stack(jacobian, dim=1)  
    for n in range(len(jacobian)):
        sing_vals = torch.linalg.svd(jacobian[n])[1]
        sing_vals = sing_vals/torch.max(sing_vals) #normalize top singular value to 1
        exp_constant = curve_fit(exp_model, np.arange(0, len(sing_vals)), sing_vals.detach(), p0=0.01)[0]
        
        ranks.append(exp_constant)
    return np.mean(ranks) #average the ranks over all samples