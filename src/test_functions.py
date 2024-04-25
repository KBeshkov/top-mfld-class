#Class of optimization functions
import numpy as np
import torch

class OptFunctions:
    def __init__(self, n_eval_points, dim, discretize=False, z_score=False):
        self.n_eval_points = n_eval_points
        self.dim = dim
        self.discretize = discretize
        self.z_score = z_score
    
    def dixon_price(self):
        x = np.random.uniform(-10,10,[self.dim,self.n_eval_points])
        f_x = (x[0]-1)**2
        for d in range(1,self.dim):
            f_x += (d+1)*(2*x[d]**2-x[d-1])**2
        if self.z_score:
            f_x = (f_x-f_x.mean())/f_x.std()
        if self.discretize:
            f_x = f_x>f_x.mean()
        return x, f_x
    
    def rosenbrock(self):
        x = np.random.uniform(-5,10,[self.dim,self.n_eval_points])
        f_x = 0
        for d in range(self.dim-1):
            f_x += (100*(x[d+1]-x[d]**2)**2+(x[d]-1)**2)
        if self.z_score:
            f_x = (f_x-f_x.mean())/f_x.std()
        if self.discretize:
            f_x = f_x>f_x.mean()
        return x,f_x
    
    def sphere(self):
        x = np.random.uniform(-5.12,5.12,[self.dim,self.n_eval_points])
        f_x = np.sum(x**2,0)
        if self.z_score:
            f_x = (f_x-f_x.mean())/f_x.std()
        if self.discretize:
            f_x = f_x>f_x.mean()
        return x,f_x
    
    def zakharov(self):
        x = np.random.uniform(-5,10,[self.dim,self.n_eval_points])
        f_x = np.sum(x**2)
        for d in range(self.dim):
            f_x += (0.5*(d+1)*x[d])**2+(0.5*(d+1)*x[d])**4
        if self.z_score:
            f_x = (f_x-f_x.mean())/f_x.std()
        if self.discretize:
            f_x = f_x>f_x.mean()
        return x,f_x
        
    def michalewicz(self,m=10):
        x = np.random.uniform(0,np.pi,[self.dim,self.n_eval_points])
        f_x = 0
        for d in range(self.dim):
            f_x += -np.sin(x[d])*(np.sin(((d+1)*x[d]**2)/np.pi)**(2*m))
        if self.z_score:
            f_x = (f_x-f_x.mean())/f_x.std()
        if self.discretize:
            f_x = f_x>f_x.mean()
        return x,f_x
    
    def schwefel(self):
        x = np.random.uniform(-500,500,[self.dim,self.n_eval_points])
        f_x = 418.9829*self.dim
        for d in range(self.dim):
            f_x += -x[d]*np.sin(np.sqrt(np.abs(x[d])))
        if self.z_score:
            f_x = (f_x-f_x.mean())/f_x.std()
        if self.discretize:
            f_x = f_x>f_x.mean()
        return x,f_x        
    
    def styblinski_tang(self):
        x = np.random.uniform(-5,5,[self.dim,self.n_eval_points])
        f_x = 0
        for d in range(self.dim):
            f_x += 0.5*(x[d]**4 - 16*x[d]**2 + 5*x[d])
        if self.z_score:
            f_x = (f_x-f_x.mean())/f_x.std()
        if self.discretize:
            f_x = f_x>f_x.mean()
        return x,f_x

def gen_classif_functions(f,n_functions):
    new_functions  = np.zeros([*np.shape(f),n_functions+1])
    new_functions[:,0] = f>=np.mean(f)
    n_vals = 2
    for n in range(n_functions):
        for k in range(n_vals+2**n):
            percentile_low = np.percentile(f.flatten(), 100*k/(n_vals+2**n))
            percentile_high = np.percentile(f.flatten(), 100*(k+1)/(n_vals+2**n))
            new_functions[np.logical_and(percentile_low<f, f<=percentile_high),n+1] = k/(n_vals+2**n)
        new_functions[:,n+1] = (new_functions[:,n+1]-np.mean(new_functions[:,n+1]))/np.std(new_functions[:,n+1]) 
        n_vals = n_vals+2**n
    return new_functions
            
            
            
            
            