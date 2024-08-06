#FROM https://github.com/fmenat/MultiviewCropClassification
import torch, copy
from torch import nn
import numpy as np
from typing import List, Union, Dict

from .utils import stack_all, object_to_list, collate_all_list, detach_all, unpack, squeeze
from .utils import get_dic_emb_dims, get_loss_by_name, map_encoders 
from .utils import get_missviews_mask_i, possible_missing_mask
from .core_fusion import _BaseViewsLightning
from src.metrics.loss import NLL
from src.models.single.base_decoders import Generic_Bayesian_Decoder

class ModelFusion(_BaseViewsLightning):
    #ONLY FOR POINT-PREDICTION
    #it is based on three modules: encoders, aggregation, prediction_head
    #only one-task (prediction) and full-view available setting
    #support list and dictionary of encoders -- but transform to dict (ideally it should be always a dict)
    #support list and dictionary of emb dims -- but transform to dict
    def __init__(self,
                 view_encoders: Union[List[nn.Module],Dict[str,nn.Module]],  #require that it contain get_output_size() .. otherwise indicate in emb_dims..
                 fusion_module: nn.Module,
                 prediction_head: nn.Module,
                 loss_args: dict ={},
                 view_names: List[str] = [], #this is only used if view_encoders are a list
                 mc_samples: int=20,
                 uq: bool=True,
                 len_sequence=1
                 ):
        super(ModelFusion, self).__init__()

        if len(view_encoders) == 0:
            raise Exception("you have to give a encoder models (nn.Module), currently view_encoders=[] or {}")
        if type(prediction_head) == type(None):
            raise Exception("you need to define a prediction_head")
        if type(fusion_module) == type(None):
            raise Exception("you need to define a fusion_module")
        if len(loss_args) == 0:
            loss_args = {"name": "mse"}
        self.save_hyperparameters(ignore=['view_encoders','prediction_head', 'fusion_module'])

        view_encoders = map_encoders(view_encoders, view_names=view_names) #view_encoders to dict if no dict yet (e.g. list)
        self.views_encoder = nn.ModuleDict(view_encoders)
        self.view_names = list(self.views_encoder.keys())
        self.fusion_module = fusion_module
        self.prediction_head = prediction_head
        self.mc_samples = mc_samples
        self.criteria = loss_args["function"] if "function" in loss_args else get_loss_by_name(**self.hparams_initial.loss_args)
        self.uq = uq # uncertainty quantification 
        self.len_sequence = len_sequence
        
    def forward_encoders(self,
            views: Dict[str, torch.Tensor],
            ) -> Dict[str, torch.Tensor]:
        if type(views) == list:
            raise Exception("Please feed forward function with dictionary data {view_name_str: torch.Tensor} instead of list")
        
        zs_views = {}
        for v_name in self.view_names:
            zs_views[v_name] = self.views_encoder[v_name](views[v_name])
        return {"views:rep": zs_views}

    def _forward(self,
            views: Dict[str, torch.Tensor],
            ) -> Dict[str, torch.Tensor]:
        if type(views) == list:
            raise Exception("Please feed forward function with dictionary data {view_name_str: torch.Tensor} instead of list")

        out_zs_views = self.forward_encoders(views) 
        out_z_e = self.fusion_module(list(out_zs_views["views:rep"].values())) #carefully, always forward a list
        out_y = self.prediction_head(out_z_e["joint_rep"])
        return {"prediction": out_y.pop("prediction"), **out_y, **out_z_e, **out_zs_views }
    
    def forward(self,
                views: Dict[str, torch.Tensor],
                sample_nbr: int=3
            ) -> Dict[str, torch.Tensor]:
        if not self.uq:  return self._forward(views=views)
        if self.training:
            nbr = sample_nbr # only few samples in training
        else: nbr = self.mc_samples
        preds = []
        vars = []
        for _ in range(nbr):
            outputs_ = self._forward(views)
            preds.append(outputs_["prediction"])
            vars.append(outputs_["variance"])
        return {"prediction": torch.stack(preds, axis=-1), "variance": torch.stack(vars, axis=-1)}

    def prepare_batch(self, batch: dict) -> list:
        views_data, views_target = batch["views"], batch["target"]

        if type(views_data) == list:
            print("views as list")
            if "view_names" in batch:
                if len(batch["view_names"]) != 0:
                    views_to_match = batch["view_names"]
            else:
                views_to_match = self.view_names #assuming a order list with view names based on the construction of the class
            views_dict = {views_to_match[i]: value for i, value in enumerate(views_data) }
        elif type(views_data) == dict:
            views_dict = views_data
        else:
            raise Exception("views in batch should be a List or Dict")

        views_target = views_target.to(torch.float32)
        return views_dict, views_target

    def loss_batch(self, batch: dict) -> dict:
        #calculate values of loss that will not return the model in a forward/transform

        views_dict, views_target = self.prepare_batch(batch)
        out_dic = self(views_dict) 
        #if self.hparams_initial.loss_args["name"].lower() == "nll":
        #    return { "objective": self.criteria(out_dic["prediction"], out_dic["variance"], views_target) }
        if isinstance(self.criteria, NLL) and self.uq:
            return {"objective": self.criteria(out_dic["prediction"], out_dic["variance"], views_target)}
        return { "objective": self.criteria(out_dic["prediction"], views_target) }
    #
    def transform(self,
            loader: torch.utils.data.DataLoader,
            device:str="",
            args_forward:dict = {},
            **kwargs
            ) -> dict:
        """
        function to get predictions from model  -- inference or testing

        :param loader: a dataloader that matches the structure of that used for training
        :return: transformed views

        #return numpy arrays based on dictionary
        """
        device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "" else device
        device_used = torch.device(device)

        self.eval() #set batchnorm and dropout off
        self.to(device_used)
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                views_dict, views_target = self.prepare_batch(batch)
                for view_name in views_dict:
                    views_dict[view_name] = views_dict[view_name].to(device_used)

                outputs_ = self(views_dict, **args_forward)
                outputs_ = squeeze(outputs_)
                #outputs_ = unpack(outputs_)
                outputs_ = detach_all(outputs_)
                if batch_idx == 0:
                    outputs = object_to_list(outputs_) #to start append values
                else:
                    collate_all_list(outputs, outputs_) #add to list in cpu
        self.train()
        return stack_all(outputs) #stack with numpy in cpu
    
    def eval(self):
        """
        Recursively sets only Batchnorm layers into eval mode to activate dropout and deactivate Batchnorm.
        """
        if not self.uq:
            return super().eval()
        def __eval(module):
            if isinstance(module, (nn.modules.batchnorm.BatchNorm1d, nn.modules.batchnorm.BatchNorm2d, nn.modules.batchnorm.BatchNorm3d, nn.modules.LayerNorm)):
                module.eval()
            else:
                module.train()
            #if isinstance(module, (Generic_Bayesian_Decoder)): module.training = False #Ensure to take multiple mc samples during eval
            for child in module.children():
                __eval(child)
        __eval(self)
        self.training = False

    def predict(self, loader: torch.utils.data.DataLoader, out_norm:bool =False, device="", **args):
        output_ = self.transform( loader, output=True, intermediate=False, out_norm=out_norm,device=device, **args)
        return output_["prediction"]