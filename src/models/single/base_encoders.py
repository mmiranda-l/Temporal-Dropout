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
        use_norm: bool = False,
        variational: bool = False,
        temp_dropout: float = 0.0, 
        temp_dropout_args : dict = {},
        **kwargs,
    ):
        super(Generic_Encoder, self).__init__()
        self.return_all = False
        self.pre_encoder = encoder

        #build encoder head
        self.latent_dims = latent_dims
        self.variational = variational
        if self.variational:
            self.linear_layer_mu = nn.Linear(self.pre_encoder.get_output_size(), self.latent_dims)
            self.linear_layer_var = nn.Linear(self.pre_encoder.get_output_size(), self.latent_dims)
        else:
            self.use_norm = use_norm
            self.linear_layer = nn.Linear(self.pre_encoder.get_output_size(), self.latent_dims) 
            self.bn_linear = nn.BatchNorm1d(self.latent_dims, affine=False) if self.use_norm else nn.Identity()
            #self.ln_linear = nn.LayerNorm(self.latent_dims, elementwise_affine=False)
        self.tempdrop_layer = TemporalDropout(p=temp_dropout,
                                           **temp_dropout_args) if temp_dropout != 0 else nn.Identity()

    def activate_return_all(self):
        self.return_all = True

    def activate_normalize_output(self):
        self.use_norm = True

    def forward(self, x):
        x = self.tempdrop_layer(x) #apply temporal dropout if available
        out_forward = self.pre_encoder(x) #should return a dictionary with output data {"rep": tensor}, or a single tensor
        if type(out_forward) != dict:
            out_forward = {"rep": out_forward}
        
        if self.variational:
            mu_ = self.linear_layer_mu(out_forward["rep"])
            logvar_ = self.linear_layer_var(out_forward["rep"])
            #sampling 
            sampled = 0 
            return_dic = {"rep": sampled,  "pre:rep:mu": mu_, "pre:rep:logvar": logvar_ }
        else:
            return_dic = {"rep": self.bn_linear(self.linear_layer(out_forward["rep"])) }

        if self.return_all:
            return_dic["pre:rep"] = out_forward.pop("rep")
            return dict(**return_dic, **out_forward)
        else:
            return return_dic["rep"] #single tensor output

    def get_output_size(self):
        return self.latent_dims
