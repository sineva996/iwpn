import torch
import torch.nn as nn
import numpy as np

def BN_freeze(model,bn_freeze,bn_affine):
    freeze_bn = bn_freeze
    freeze_bn_affine = bn_affine
    if freeze_bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m,nn.BatchNorm1d):
                m.eval()
                if freeze_bn_affine:
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
    return model

class Differ_freeze(nn.Module):
    def __init__(self,model,iter):
        super(Differ_freeze, self).__init__()
        self.frozen_layer=[]
        self.iter=iter
        self.model=model

    def get_frozen_layer(self):
        if self.iter==1:
            frozen_layer=[self.model.layer1]
        elif self.iter==2:
            frozen_layer=[self.model.layer1,self.model.layer2]
        else:
            print("error")
        
        return frozen_layer
        
    def getmodel(self):
        self.frozen_layer= self.get_frozen_layer()
        
        for layer in self.frozen_layer:
            for name, value in layer.named_parameters():
                value.requires_grad = False
        
        return self.model

    def model_params(self):
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        return params
"""The Key of the mask:
Conv1Weight
Layer1.0.Conv1Weight
Layer1.0.Conv2Weight
Layer1.1.Conv1Weight
Layer1.1.Conv2Weight
Layer1.2.Conv1Weight
Layer1.2.Conv2Weight
Layer2.0.Conv1Weight
Layer2.0.Conv2Weight
Layer2.0.Downsample.0Weight
Layer2.1.Conv1Weight
Layer2.1.Conv2Weight
Layer2.2.Conv1Weight
Layer2.2.Conv2Weight
Layer2.3.Conv1Weight
Layer2.3.Conv2Weight
Layer3.0.Conv1Weight
Layer3.0.Conv2Weight
Layer3.0.Downsample.0Weight
Layer3.1.Conv1Weight
Layer3.1.Conv2Weight
Layer3.2.Conv1Weight
Layer3.2.Conv2Weight
Layer3.3.Conv1Weight
Layer3.3.Conv2Weight
Layer3.4.Conv1Weight
Layer3.4.Conv2Weight
Layer3.5.Conv1Weight
Layer3.5.Conv2Weight
Layer4.0.Conv1Weight
Layer4.0.Conv2Weight
Layer4.0.Downsample.0Weight
Layer4.1.Conv1Weight
Layer4.1.Conv2Weight
Layer4.2.Conv1Weight
Layer4.2.Conv2Weight
Fcweight"""
def frozen_mask(mask,model,itera):
    prunable_types=(nn.Conv2d, nn.Linear)
    if itera==1:
        for mod_name, mod in model.named_modules():
            if isinstance(mod, prunable_types):
                for name, param_layer in mod.named_parameters():
                    if ('bias' not in name) and (mod_name == 'conv1'):
                        prev_mask = torch.ones(param_layer.size(), dtype=torch.bool, requires_grad=False)
                        mask[mod_name+name]=prev_mask
                    if ('bias' not in name) and ('layer1' in mod_name):
                        prev_mask = torch.ones(param_layer.size(), dtype=torch.bool, requires_grad=False)
                        mask[mod_name+name]=prev_mask
                    """ if ('bias' not in name) and ('Fc' in mod_name):
                        prev_mask = torch.ones(param_layer.size(), dtype=torch.bool, requires_grad=False)
                        mask[mod_name+name]=prev_mask """
    elif itera==2:
        for mod_name, mod in model.named_modules():
            if isinstance(mod, prunable_types):
                for name, param_layer in mod.named_parameters():
                    if ('bias' not in name) and (mod_name == 'conv1'):
                        prev_mask = torch.ones(param_layer.size(), dtype=torch.bool, requires_grad=False)
                        mask[mod_name+name]=prev_mask
                    if ('bias' not in name) and (('layer1' in mod_name)or('layer2' in mod_name)):
                        prev_mask = torch.ones(param_layer.size(), dtype=torch.bool, requires_grad=False)
                        mask[mod_name+name]=prev_mask
                    """ if ('bias' not in name) and ('Fc' in mod_name):
                        prev_mask = torch.ones(param_layer.size(), dtype=torch.bool, requires_grad=False)
                        mask[mod_name+name]=prev_mask """
    
def get_feat_params(model,iter):
    if iter==0:
        params=list(model.conv1.parameters())+list(model.layer1.parameters())+list(model.layer2.parameters())+list(model.layer3.parameters())+list(model.layer4.parameters())
    elif iter==1:
        params=list(model.layer2.parameters())+list(model.layer3.parameters())+list(model.layer4.parameters())
    elif iter==2:
        params=list(model.layer3.parameters())+list(model.layer4.parameters())
    return params

def get_fc_params(model):
    params=list(model.fc.parameters())
    return params