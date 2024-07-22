import torch
import torch.nn as nn

class DropoutCore(nn.Module):
    def __init__(self):
        super(DropoutCore, self).__init__()

    def eval(self):
        """
        Only Batchnorm layers into eval mode to activate dropout and deactivate Batchnorm
        """
        for module in self.modules():
            if isinstance(module, (nn.modules.BatchNorm1d , nn.modules.BatchNorm2d, nn.modules.BatchNorm3d )):
                module.eval()
            else: module.train()
        self.training = False