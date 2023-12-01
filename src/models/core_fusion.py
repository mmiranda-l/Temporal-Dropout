import pytorch_lightning as pl
import torch, copy
from torch import nn
import numpy as np

from .utils import count_parameters 


class _BaseViewsLightning(pl.LightningModule):
    def __init__(
            self,
            optimizer="adam",
            lr=1e-3,
            weight_decay=0,
            extra_optimizer_kwargs=None,
            lr_decay_steps=None,
    ):
        super().__init__()
        if extra_optimizer_kwargs is None:
            extra_optimizer_kwargs = {}
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.extra_optimizer_kwargs = extra_optimizer_kwargs

    def training_step(self, batch, batch_idx):
        """
            batch sould be a dictionary containin key 'views' for data and 'target' for the desired output to learn
        """
        loss = self.loss_batch(batch)
        for k, v in loss.items():
            self.log("train_" + k, v, prog_bar=True)
        return loss["objective"]

    def validation_step(self, batch, batch_idx):
        """
            batch sould be a dictionary containin key 'views' for data and 'target' for the desired output to learn
        """
        loss = self.loss_batch(batch)
        for k, v in loss.items():
            self.log("val_" + k, v)
        return loss["objective"]

    def configure_optimizers(self):
        """Collects learnable parameters and configures the optimizer and learning rate scheduler.
        Returns:
            Tuple[List, List]: two lists containing the optimizer and the scheduler.
        """
        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam
        elif self.optimizer == "adamw":
            optimizer = torch.optim.AdamW
        elif self.optimizer == "lbfgs":
            optimizer = torch.optim.LBFGS
        else:
            raise ValueError(f"{self.optimizer} not in (sgd, adam, adamw)")
        return  optimizer(self.parameters(),lr=self.lr, weight_decay=self.weight_decay, **self.extra_optimizer_kwargs)

    def count_parameters(self):
        return count_parameters(self)