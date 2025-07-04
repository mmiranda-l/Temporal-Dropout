import torch
from torch import nn
import abc

from .temporal_dropout import TemporalDropout

class Base_Encoder(abc.ABC, nn.Module):
    """
    Class to add methods for common modality specific methods
    """

    @abc.abstractmethod
    def get_output_size(self):
        pass


class Generic_Encoder(Base_Encoder):
    """
        it adds a linear layer at the end of an encoder model with possible batch normalization.
        The linear layer could be variational in some extension
    """
    def __init__(
        self,
        encoder: nn.Module,
        latent_dims: int,
        temp_dropout: float = 0.0, 
        temp_dropout_args : dict = {},
        **kwargs,
    ):
        super(Generic_Encoder, self).__init__()
        self.pre_encoder = encoder

        #build encoder head
        self.latent_dims = latent_dims
        self.linear_layer = nn.Linear(self.pre_encoder.get_output_size(), self.latent_dims) 
        
        self.tempdrop_layer = TemporalDropout(p=temp_dropout,
                                           **temp_dropout_args) if temp_dropout != 0 else nn.Identity()

    def forward(self, x):
        x = self.tempdrop_layer(x) #apply temporal dropout if available
        
        out_forward = self.pre_encoder(x) 
        if type(out_forward) == dict:
            final_rep = self.linear_layer(out_forward["rep"])
        else:
            final_rep = self.linear_layer(out_forward)
        
        return {
            "rep": final_rep,
            }

    def get_output_size(self):
        return self.latent_dims
