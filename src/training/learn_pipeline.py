import shutil, os, sys, gc, time
from typing import List, Union, Dict
from pathlib import Path
import copy
import numpy as np

import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers import MLFlowLogger
import pytorch_lightning as pl

from src.models.fusion import InputFusion
from src.models.network_models import create_model
from src.datasets.utils import _to_loader


def prepare_loggers(data_name, method_name, run_id, fold_id, folder_c, tags_ml, run_id_mlflow,monitor_name, **early_stop_args):
    save_dir_tsboard = f'{folder_c}/tensorboard_logs/'
    save_dir_chkpt = f'{folder_c}/checkpoint_logs/'
    save_dir_mlflow = f'{folder_c}/mlruns/'
    exp_folder_name = f'{data_name}/{method_name}'

    for v in Path(f'{save_dir_chkpt}/{exp_folder_name}/').glob(f'r={run_id:02d}_{fold_id:02d}*'):
        v.unlink()
    if os.path.exists(f'{save_dir_tsboard}/{exp_folder_name}/version_{run_id:02d}_{fold_id:02d}'):
        shutil.rmtree(f'{save_dir_tsboard}/{exp_folder_name}/version_{run_id:02d}_{fold_id:02d}')
    early_stop_callback = EarlyStopping(monitor=monitor_name, **early_stop_args)
    tensorlogg = TensorBoardLogger(name="", save_dir=f'{save_dir_tsboard}/{exp_folder_name}/')
    checkpoint_callback = ModelCheckpoint(monitor=monitor_name, mode=early_stop_args["mode"], every_n_epochs=1, save_top_k=1,
        dirpath=f'{save_dir_chkpt}/{exp_folder_name}/', filename=f'r={run_id:02d}_{fold_id:02d}-'+'{epoch}-{step}-{val_objective:.2f}')
    tags_ml = dict(tags_ml,**{"data_name":data_name,"method_name":method_name})
    if run_id_mlflow == "ind":
        mlf_logger = MLFlowLogger(experiment_name=exp_folder_name, run_name = f"version-{run_id:02d}_{fold_id:02d}",
                              tags = tags_ml, tracking_uri=f"file:{save_dir_mlflow}")
    else:
        mlf_logger = MLFlowLogger(experiment_name=data_name, run_name = method_name,
                              run_id= run_id_mlflow,
                              tags = tags_ml, tracking_uri=f"file:{save_dir_mlflow}")
    return {"callbacks": [early_stop_callback,checkpoint_callback], "loggers":[mlf_logger,tensorlogg]}

def log_additional_mlflow(mlflow_model, trainer, model):
    mlflow_model.experiment.log_artifact(mlflow_model.run_id, trainer.checkpoint_callback.best_model_path, artifact_path="models")
    mlflow_model.experiment.log_text(mlflow_model.run_id, str(model), "models/model_summary.txt")
    mlflow_model.experiment.log_dict(mlflow_model.run_id, model.count_parameters(), "models/model_parameters.yaml")
        
def InputFusion_train(train_data: dict, val_data = None,
                data_name="", method_name="", run_id=0, fold_id=0, output_dir_folder="", run_id_mlflow=None,
                training={}, architecture= {}, **kwargs):
    emb_dim = training["emb_dim"]
    max_epochs = training["max_epochs"]
    batch_size = training["batch_size"]
    early_stop_args = training["early_stop_args"]
    loss_args = training["loss_args"]
    folder_c = output_dir_folder+"/run-saves"
    n_labels = 1

    #MODEL DEFINITION
    feats_dims = [v.shape[-1] for v in train_data["views"]]
    args_model = {"input_dim_to_stack": feats_dims, "loss_args": loss_args}

    encoder_model = create_model(np.sum(feats_dims), emb_dim, **architecture["encoders"])
    predictive_model = create_model(emb_dim, n_labels, **architecture["predictive_model"], encoder=False)  #default is mlp
    full_model = torch.nn.Sequential(encoder_model, predictive_model)
    #FUSION DEFINITION
    model = InputFusion(predictive_model=full_model, view_names=train_data["view_names"], **args_model)
    
    #DATA DEFITNION
    if type(val_data) != type(None):
        val_dataloader = _to_loader(val_data, batch_size=batch_size, train=False)
        monitor_name = "val_objective"
    else:
        monitor_name = "train_objective"
    train_dataloader = _to_loader(train_data, batch_size=batch_size)
    extra_objects = prepare_loggers(data_name, method_name, run_id, fold_id, folder_c, model.hparams_initial, run_id_mlflow, monitor_name, **early_stop_args)
    device = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator=device, devices = 1,
                         callbacks=extra_objects["callbacks"],logger=extra_objects["loggers"])
    trainer.fit(model, train_dataloader, val_dataloaders=(val_dataloader if type(val_data) != type(None) else None))

    checkpoint = torch.load(trainer.checkpoint_callback.best_model_path)
    model.load_state_dict(checkpoint['state_dict'])
    log_additional_mlflow(trainer.loggers[0], trainer, model)
    return model, trainer