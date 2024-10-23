
import torch
from torch import nn
import numpy as np

class ConcreteTemporalDropout(nn.Module):
    def __init__(self, 
                 weight_regularizer=1e-6,
                 dropout_regularizer=1e-5, 
                 init_min=0.01, 
                 init_max=0.5, 
                 ):
        super(ConcreteTemporalDropout, self).__init__()

        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        
        init_min = np.log(init_min) - np.log(1. - init_min)
        init_max = np.log(init_max) - np.log(1. - init_max)
        
        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max))
        
    def forward(self, x):
        p = torch.sigmoid(self.p_logit)
        out = self._concrete_dropout(x, p)

        #weights_regularizer = self.weight_regularizer * sum_of_square / (1 - p)
        
        dropout_regularizer = p * torch.log(p)
        dropout_regularizer += (1. - p) * torch.log(1. - p)
        
        input_dimensionality = x[0].numel() # Number of elements of first item in batch
        dropout_regularizer *= self.dropout_regularizer * input_dimensionality #used for loss 
        #return out, p
        return out
        
    def _concrete_dropout(self, x, p):
        """X must be a tensor with a temporal dimension of shape B X T X C"""
        eps = 1e-7
        temp = 0.1

        unif_noise = torch.rand_like(x[:,0,:]) # Take only batch and temporal dimension

        drop_prob = (torch.log(p + eps)
                    - torch.log(1 - p + eps)
                    + torch.log(unif_noise + eps)
                    - torch.log(1 - unif_noise + eps))
        
        drop_prob = torch.sigmoid(drop_prob / temp)
        random_tensor = 1 - drop_prob
        retain_prob = 1 - p
        random_tensor = random_tensor.unsqueeze(2).repeat(1, 1, x.shape[1]).permute(0, 2, 1)
        x  = torch.mul(x, random_tensor)
        x /= retain_prob
        return x