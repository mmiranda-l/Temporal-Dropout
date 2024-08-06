from torch import nn
from typing import List, Union, Dict

from .base_fusion import ModelFusion
from .fusion_module import FusionModule

class InputFusion(ModelFusion):
    def __init__(self,
                 predictive_model,
                 fusion_module: dict = {},
                 loss_args: dict = {},
                 view_names: List[str] = [],
                 input_dim_to_stack: Union[List[int], Dict[str,int]] = 0,
                 uq: bool=True,
                 len_sequence=1,
                 ):
        if len(loss_args) == 0:
            loss_args = {"name": "mse"}
        if type(fusion_module) == dict:
            if len(fusion_module) == 0:
                fusion_module = {"mode": "concat", "adaptive":False, "emb_dims": input_dim_to_stack }
            fusion_module = FusionModule(**fusion_module)
        fake_view_encoders = []
        for v in fusion_module.emb_dims:
            aux = nn.Identity()
            aux.get_output_size = lambda : v
            fake_view_encoders.append( aux)
        super(InputFusion, self).__init__(fake_view_encoders, fusion_module, predictive_model,
            loss_args=loss_args, view_names=view_names, uq=uq, len_sequence=len_sequence)