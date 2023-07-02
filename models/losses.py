import torch.nn as nn
import torch
from util.util import get_module_by_name

class BNSLoss(nn.Module):
    def __init__(self, model):
        self.means = {}
        self.vars = {}
        self.names = []
        for name, module in self.model.named_modules():
            if 'norm' in name:
                mean, var = self.get_statistics(module)
                self.means[name] = mean
                self.vars[name] = var
                self.names.append(name)
    
    def foward(self, model):
        loss_mean = 0
        loss_var = 0

        for name in names:
            module = get_module_by_name(model, name)
            fixed_mean = self.means[name]
            fixed_var = self.vars[name]
            loss_mean += torch.norm(module.running_mean - fixed_mean, 2).mean()
            loss_var += torch.norm(module.running_var - fixed_var, 2).mean()
        
        return loss_mean.mean(), loss_var.mean()
        
class MaxSqaureLoss(nn.Module):

    def __init__(self,):
        pass
    
    def forward(self, p):
        p = -torch.mean(torch.sum(torch.pow(p, 2), dim=1))
        return p

        