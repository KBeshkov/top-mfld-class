import torch
import torch.nn as nn
import tqdm
import sys
sys.path.append('/Users/kosio/Repos/LUP-rank-computer/')
from LUP_rank import rank_revealing_LUP
import numpy as np
from scipy.optimize import curve_fit


class FeedforwardNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, activation=nn.ReLU, out_layer_sz = 0, mean=-0.5, std=2, init_type = 'uniform',conv_dict={}):
        super(FeedforwardNetwork, self).__init__()


        #decompositoins
        self.global_decomposition = []
        self.local_decomposition = []
        
        self.layers = nn.ModuleList()
        self.activation = activation
        
        self.mean = mean
        self.std = std
        
        self.input_size = input_size
        self.init_type = init_type
        self.out_layer_sz = out_layer_sz
        
        self.hidden_sizes = hidden_sizes

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
    
    def get_map_at_point(self,x, layer_num):
        layer_n = 0
        for n, layer in enumerate(self.layers):
            x = layer(x.flatten())
            if isinstance(layer, nn.Linear):
                out_sign = (x>0).float()
                Q = (torch.eye(layer.weight.shape[0])*out_sign)
                if n==0:
                    out_mat = Q@layer.weight
                    # out_bias[n] = Q@layer.bias
                else:
                    out_mat = Q@layer.weight@out_mat
                if layer_n==layer_num:
                    return out_mat
                layer_n+=1
        return out_mat#, out_bias

                    
    def get_sign_complex(self, layers=1):
        #Generate the sign complex as defined in: Algorithmic Determination of the Combinatorial Structure of the
        #Linear Regions of ReLU Neural Networks, https://arxiv.org/abs/2207.07696
        cubic_points = []
        for n, layer in enumerate(self.layers):
            if n==0:
                combinations = torch.combinations(torch.arange(0,self.hidden_sizes[0]),r=self.input_size)
                layer_weights = self.layers[0].weight
                layer_bias = self.layers[0].bias
                for c in combinations:
                    intersect_points = torch.linalg.solve(layer_weights[c].detach(),-layer_bias[c].detach())
                    cubic_points.append(intersect_points)
            elif isinstance(layer, nn.Linear):
                combinations = torch.combinations(self.hidden_sizes[n+1].detach(),r=self.hidden_sizes[n])
        return torch.vstack(cubic_points)

    
    def compute_rank(self, out_sign):
        out_mat = 0
        ranks = []
        for x in out_sign:
            for layer in self.layers:
                if isinstance(layer, nn.Linear):    
                    Q = (torch.eye(layer.weight.shape[0])*x)
                    out_mat = Q@layer.weight
            try:
                fin_rank = rank_revealing_LUP(out_mat.detach().numpy().astype('float')) #change from torch.matrix_rank for speedup
            except:
                fin_rank = torch.linalg.matrix_rank(out_mat)
            ranks.append(fin_rank)
        return ranks
    
    def compute_rank_at_point(self, x):
        ranks = []
        if len(self.conv_dict)>0:
            x = self.feature_layer(x)
        for n in range(len(self.hidden_sizes)+1):
            out_mat = self.get_map_at_point(x,layer_num=n)
            # try:
                # fin_rank = rank_revealing_LUP(out_mat.detach().numpy().astype('float'))
            # except:
            fin_rank = torch.linalg.matrix_rank(out_mat)
            ranks.append(fin_rank)
        return ranks
    
    def compute_rank_classes(self,X):
        final_rankwords = np.zeros([len(self.hidden_sizes)+1,len(X)])
        for n in range(len(X)):
            final_rankwords[:,n] = self.compute_rank_at_point(X[n])#[:-1]
        eq_classes = [[] for i in range(len(self.hidden_sizes)+1)]#np.zeros([len(self.hidden_sizes),len(X)])
        for i in range(len(self.hidden_sizes)+1):
            _, inverse_indices = np.unique(final_rankwords[i], return_inverse=True)
            for n in np.unique(inverse_indices):
                entries = np.where(inverse_indices==n)[0]
                if len(entries)>0:
                    eq_classes[i].append(entries)
        return eq_classes, final_rankwords
    
    def compute_codeword_at_point(self, x, top_layer=-1, mode='global'):
        codeword = []
        if len(self.conv_dict)>0:
            x = self.feature_layer(x)
        layer_counter = 0
        if mode=='global':
            for n, layer in enumerate(self.layers):
                x = layer(x.flatten())
                if isinstance(layer, nn.Linear):
                    if n ==len(self.layers):
                        out_sign = map(str,list(np.ones(len(x)).astype(np.int64)))
                    else:
                        out_sign = map(str,list((x>0).detach().numpy().astype(np.int64)))
                    codeword.append(''.join(out_sign))
                    layer_counter+=1
                    if layer_counter==top_layer:
                        return ''.join(codeword)
        if mode=='local':
            for n, layer in enumerate(self.layers):
                x = layer(x.flatten())
                if isinstance(layer, nn.Linear):
                    if n ==len(self.layers):
                        out_sign = map(str, list(np.ones(len(x)).astype(np.int64)))
                    else:
                        out_sign = map(str,list((x>0).detach().numpy().astype(np.int64)))
                    layer_counter+=1
                    if layer_counter==top_layer:
                        codeword=out_sign
                        return ''.join(codeword)
    
    def compute_codeword_eq_classes(self, X, top_layer=-1, mode='global'):
        final_codewords = np.empty([len(self.hidden_sizes)+1,len(X)], dtype=object)#.astype(np.int64)
        for n in range(len(X)):
            for i in range(len(self.hidden_sizes)+1):
                final_codewords[i,n] = self.compute_codeword_at_point(X[n],top_layer=i+1,mode=mode)                    
        _, inverse_indices = np.unique(final_codewords, return_inverse=True)
        eq_classes = [[] for i in range(len(self.hidden_sizes)+1)]
        for i in range(len(self.hidden_sizes)+1):
            _, inverse_indices = np.unique(final_codewords[i], return_inverse=True)
            for n in np.unique(inverse_indices):
                entries = np.where(inverse_indices==n)[0]
                if len(entries)>0:
                    eq_classes[i].append(entries)
        if mode=='global':
            self.global_decomposition = [eq_classes, final_codewords]
        elif mode=='local':
            self.local_decomposition = [eq_classes, final_codewords]
        return eq_classes, final_codewords
    
    def find_overlapping_eq_classes(self, X, layer=-1, sensitivity=1e-6,brute_force=False):
        uf = UnionFind()
        if len(self.local_decomposition)==0:
            local_classes = self.compute_codeword_eq_classes(X,mode='local')[0][layer]
        else:
            local_classes = self.local_decomposition[0][layer]
        if len(self.global_decomposition)==0:
            global_classes = self.compute_codeword_eq_classes(X)[0][layer]
        else:
            global_classes = self.global_decomposition[0][layer]
        class_partitions = [[] for i in range(len(local_classes))]
        all_classes = []
        contractible_classes = []
        # for i, loc_class in enumerate(local_classes):
        # for glob_class in global_classes:
            # if set(glob_class).issubset(loc_class):
                # class_partitions[i].append(glob_class)
        for n, class_n in enumerate(global_classes):#class_partitions[i]):
            all_classes.append(list(class_n))
            for m, class_m in enumerate(global_classes):#class_partitions[i]):
                if n>m:
                    intersect = self.find_intersection_simple(X[class_n],X[class_m],layer,epsilon=sensitivity, brute_force=brute_force)
                    overlap_class = [[class_n[intersect[i,0]], class_m[intersect[i,1]]] for i in range(len(intersect))]
                    all_classes.extend(overlap_class)
                    contractible_classes.extend(overlap_class)
                # A = self.get_map_at_point(X[class_n[0]], layer=layer).detach().numpy()
                # B = self.get_map_at_point(X[class_m[0]], layer=layer).detach().numpy()
                # if len(class_n)<100000 and len(class_n)<100000:
                #     intersect = find_intersection(A, B, X[class_n].detach().numpy(), X[class_m].detach().numpy(),epsilon=sensitivity,brute_force=True)
                # else:
                #     intersect = find_intersection(A, B, X[class_n].detach().numpy(), X[class_m].detach().numpy(),epsilon=sensitivity,brute_force=False)
                # if intersect:
                #     all_classes.append(list(np.concatenate([class_n, class_m])))
                #     contractible_classes.append(list(np.concatenate([class_n, class_m])))
        return uf.merge_lists(contractible_classes), uf.merge_lists(all_classes)

    def find_intersection_simple(self,C1,C2,layer=-1,epsilon=1e-6,brute_force=True):
        F1 = self.forward(C1)[layer].detach()
        F2 = self.forward(C2)[layer].detach()
        if brute_force: #maybe make threshold relative to the distance in the input space?
            A1 = self.get_map_at_point(C1[0], layer)
            A2 = self.get_map_at_point(C2[0], layer)
            epsilon = (epsilon*max(max(torch.linalg.svd(A1)[1]),max(torch.linalg.svd(A2)[1]))).item()
            d_out  = cdist(F1,F2)
            if torch.all(A1==A2):
                np.fill_diagonal(d_out,2*epsilon+1e-12)
            contract_mask = np.argwhere((d_out)<=epsilon)
            return contract_mask
        else:
            for p in F2:
                c = np.zeros(len(F1))
                A = np.r_[F1.T,np.ones((1,len(F1)))]
                b = np.r_[p, np.ones(1)]
                # b_up = np.r_[p, np.ones(1)+epsilon]
                # b_low = np.r_[p, np.ones(1)-epsilon]
                bounds = [(0, None) for _ in range(len(F1))]
        
                # Solve the linear programming problem
                res = linprog(c, A_eq=A, b_eq=b, bounds=bounds, method='highs')
                if res.success:
                    # print(lp.fun)
                    return True  
                # res = linprog(c, A_eq=A, b_eq=b_low, bounds=bounds, method='highs')
                # if res.success:
                #     # print(lp.fun)
                #     return True  
                
            for p in F1:
                c = np.zeros(len(F2))
                A = np.r_[C2.T,np.ones((1,len(F2)))]
                b = np.r_[p, np.ones(1)]
                # b_up = np.r_[p, np.ones(1)+epsilon]
                # b_low = np.r_[p, np.ones(1)-epsilon]
                bounds = [(0, None) for _ in range(len(F2))]
        
                # Solve the linear programming problem
                res = linprog(c, A_eq=A, b_eq=b, bounds=bounds, method='highs')
                if res.success:
                    # print(lp.fun)
                    return True  
                # res = linprog(c, A_eq=A, b_eq=b_low, bounds=bounds, method='highs')
                # if res.success:
                #     # print(lp.fun)
                #     return True  
            return False
    
    
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

#Union find algorithm to concatenate jointly overlapping sets
class UnionFind:
    def __init__(self):
        self.parent = {}
        self. rank = {}
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x != root_y:
            if self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            elif self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1
                
    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
    
    def merge_lists(self, lists):
        for lst in lists:
            if lst:
                first_element = lst[0]
                self.add(first_element)
                
            for elem in lst[1:]:
                self.add(elem)
                self.union(first_element, elem)
        components = {}
        for lst in lists:
            for elem in lst:
                root = self.find(elem)
                if root not in components:
                    components[root] = set()
                components[root].update(lst)

        return [list(comp) for comp in components.values()]
                    
       
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

from scipy.optimize import linprog
from scipy.spatial.distance import cdist
# import cvxpy

def distance_function(x, A, B, C1, C2):
    """Objective function: minimize the distance between A*v and B*w."""
    # Extract v and w from the optimization variable x
    v = x[:len(C1[0])]  # Vector v from C1
    w = x[len(C1[0]):]  # Vector w from C2
    
    # Apply the transformations A and B
    Av = np.dot(A, v)
    Bw = np.dot(B, w)
    
    # Compute Euclidean distance between Av and Bw
    return np.linalg.norm(Av - Bw)

