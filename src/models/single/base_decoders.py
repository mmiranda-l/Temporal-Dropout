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