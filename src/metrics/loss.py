import torch.nn as nn
import torch

class NLL(nn.Module):
    def __init__(self): 
        super(NLL, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, prediction, variance, target, epsilon=1e-8):
        """
        Gaussian negative log-likelihood for regression, with variance estimated by the model.
        This function returns a keras regression loss, given a symbolic tensor for the sigma square output of the model.
        The training model should return the mean, while the testing/prediction model should return the mean and variance.

        Arguments:
        ----------
            prediction: dict
            target: tensor
        """
        #assert prediction.ndim >= 3 and target.dim() == 2 # if b x mc x p, b x 1
        loss = 0
        #for mu, var in prediction.permute(-1, 0, 1, 2): # nbr, (mu, logvar), batchsize 
        #for pre in prediction.permute(-1, 0, 1): 
            #loss += self._nll(mu, var, target, epsilon)
            #loss += self.mse(pre, target)
        mc_samples = prediction.shape[-1]
        for i in range(mc_samples):
            loss += self._nll(prediction[:,:,i], variance[:,:,i], target)
        return loss / mc_samples
    
    
    
    def _nll(self, mu, var, target, epsilon=1e-8):
        sigma = torch.exp(var)   
        return 0.5 * torch.mean(torch.log(sigma + epsilon) + torch.square(target - mu) / (sigma + epsilon))
            

if __name__ =="__main__":
    t = torch.randn(10)
    p = torch.rand(10)
    l = torch.rand(10)
    nll = NLL()
    loss =nll({"logvar": l, "prediction": p}, t)   
    print(loss)