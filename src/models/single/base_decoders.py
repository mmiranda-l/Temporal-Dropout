import torch
from torch import nn
import abc

class Base_Decoder(abc.ABC, nn.Module):
    """
    Class to add methods for common modality specific methods
    """

    @abc.abstractmethod
    def get_output_size(self):
        pass

class Generic_Decoder(Base_Decoder):
    """
        it adds a prediction head (linear layer) with possible batch normalization to decoder layers.
    """
    def __init__(
        self,
        decoder: nn.Module,
        out_dims: int,
        **kwargs,

    ):
        super(Generic_Decoder, self).__init__()
        self.pre_decoder = decoder
        #build decoder head
        self.out_dims = out_dims
        self.linear_layer = nn.Linear(self.pre_decoder.get_output_size(), self.out_dims)

    def forward(self, x):
        if type(x) == dict:
            out_forward = self.pre_decoder(x["rep"])
        else:
            out_forward = self.pre_decoder(x) 
        
        if type(out_forward) == dict:
            final_rep = self.linear_layer(out_forward["rep"])
        else:
            final_rep = self.linear_layer(out_forward)
            out_forward = {} #for returning 
        return {
            "prediction": final_rep, 
            **out_forward 
            }
        
    def get_output_size(self):
        return self.out_dims
    
class Generic_Bayesian_Decoder(Generic_Decoder):
    def __init__(self, decoder: nn.Module, out_dims: int, mc_samples: int=20, **kwargs):
        super().__init__(decoder, out_dims, **kwargs)
        self.linear_var = nn.Linear(self.pre_decoder.get_output_size(), self.out_dims) # second prediction head to model variance
        self.mc_samples = mc_samples

    def _forward(self, x):
        if type(x) == dict:
            out_forward = self.pre_decoder(x["rep"])
        else:
            out_forward = self.pre_decoder(x) 
        if type(out_forward) == dict:
            final_rep = self.linear_layer(out_forward["rep"])
            final_var = self.linear_var(out_forward["rep"])
        else:
            final_rep = self.linear_layer(out_forward)
            final_var = self.linear_var(out_forward)
            out_forward = {} #for returning 
        return final_rep, final_var
    
    def forward(self, x, sample_nbr=3):
        if self.training:
            nbr = sample_nbr # only few samples in training
        else: nbr = self.mc_samples
        preds = []
        vars = []
        for _ in range(nbr):
            pred, var = self._forward(x)
            preds.append(pred)
            vars.append(var)
        return {"prediction": torch.stack(preds, axis=-1), "variance": torch.stack(vars, axis=-1)}

        
    def get_output_size(self):
        return self.out_dims


