#FROM https://github.com/fmenat/MultiviewCropClassification
import torch, copy
from torch import nn
import numpy as np
from typing import List, Union, Dict

from .utils import stack_all, object_to_list, collate_all_list, detach_all
from .utils import get_dic_emb_dims, get_loss_by_name, map_encoders 
from .utils import get_missviews_mask_i, possible_missing_mask
from .core_fusion import _BaseViewsLightning

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

        self.criteria = loss_args["function"] if "function" in loss_args else get_loss_by_name(**self.hparams_initial.loss_args)

    def forward_encoders(self,
            views: Dict[str, torch.Tensor],
            ) -> Dict[str, torch.Tensor]:
        if type(views) == list:
            raise Exception("Please feed forward function with dictionary data {view_name_str: torch.Tensor} instead of list")
        
        zs_views = {}
        for v_name in self.view_names:
            zs_views[v_name] = self.views_encoder[v_name](views[v_name])
        return {"views:rep": zs_views}

    def forward(self,
            views: Dict[str, torch.Tensor],
            ) -> Dict[str, torch.Tensor]:
        if type(views) == list:
            raise Exception("Please feed forward function with dictionary data {view_name_str: torch.Tensor} instead of list")
        
        out_zs_views = self.forward_encoders(views) 
        out_z_e = self.fusion_module(list(out_zs_views["views:rep"].values())) #carefully, always forward a list
        
        out_y = self.prediction_head(out_z_e["joint_rep"])
        return {"prediction": out_y.pop("prediction"), **out_y, **out_z_e, **out_zs_views }
            
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

        return { "objective": self.criteria(out_dic["prediction"], views_target) }

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
                outputs_ = detach_all(outputs_)
                if batch_idx == 0:
                    outputs = object_to_list(outputs_) #to start append values
                else:
                    collate_all_list(outputs, outputs_) #add to list in cpu
        self.train()
        return stack_all(outputs) #stack with numpy in cpu

    def predict(self, loader: torch.utils.data.DataLoader, out_norm:bool =False, device="", **args):
        return self.transform( loader, output=True, intermediate=False, out_norm=out_norm,device=device, **args)["prediction"]
