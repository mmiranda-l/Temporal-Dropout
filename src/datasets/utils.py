import numpy as np

from typing import List, Union, Dict
from src.datasets.views_structure import DataViews #to store data
from .views_loader import DataViews_torch

def _to_loader(data: Union[List[np.ndarray],Dict[str,np.ndarray]], batch_size=32, train=True , args_loader={}, **args_structure):
    if type(data) == dict:
        aux_str = DataViews_torch(**data, **args_structure, train=train)
    else:
        aux_str = DataViews_torch(data, **args_structure, train=train)
    return aux_str.get_torch_dataloader(batch_size = batch_size, **args_loader)


STATIC_FEATS = ["soil", "dem", "lidar", "landcover"]

def codify_labels(array, labels):
    labels_2_idx = {v: i for i, v in enumerate(labels)} 
    return np.vectorize(lambda x: labels_2_idx[x])(array)

def reverse_padding(array, pad_val = np.nan):
    n= len(array)
    mask_nans = np.isnan(array)
    new_array= np.pad(array[~mask_nans], (np.sum(mask_nans),0), mode="constant", constant_values=pad_val)
    return new_array
    
def storage_set(data, path_out_dir, mode = "input", name="lfmc", target_names=[]):
	index, view_data, target = data
	print({"Index": len(index), "Target": len(target)}, {v: len(view_data[v]) for v in view_data})

	data_views = DataViews()
	for view_name in view_data:
		if mode.lower() == "input" and view_name.lower() in STATIC_FEATS: 
			data = view_data[view_name]
		else:
			if view_name.lower() in STATIC_FEATS:
				data = view_data[view_name][:,0,:]
			else:
				data = view_data[view_name]
			
		data_views.add_view(data, identifiers=index, name=view_name)
	data_views.add_target(target, identifiers=index,target_names=target_names)

	print(f"data stored in {path_out_dir}/{name}_train")
	add_info = "_input" if mode == "input" else ""
	data_views.save(f"{path_out_dir}/{name}_train{add_info}", xarray=True)