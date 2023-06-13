import torch
import torch.nn as nn
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_initmask(model):
    prunable_types=(nn.Conv2d, nn.Linear)
    mask={}
    for mod_name, mod in model.named_modules():
        if isinstance(mod,prunable_types):
            for name, param_layer in mod.named_parameters():
                if 'bias' not in name:
                    prev_mask = torch.zeros(param_layer.size(), dtype=torch.bool, requires_grad=False)
                    mask[mod_name+name]=prev_mask
    return mask
"""The Key of the mask:
    Conv1Weight
    Layer1.0.Conv1Weight
    Layer1.0.Conv2Weight
    Layer1.1.Conv1Weight
    Layer1.1.Conv2Weight
    Layer2.0.Conv1Weight
    Layer2.0.Conv2Weight
    Layer2.0.Downsample.0Weight
    Layer2.1.Conv1Weight
    Layer2.1.Conv2Weight
    Layer3.0.Conv1Weight
    Layer3.0.Conv2Weight
    Layer3.0.Downsample.0Weight
    Layer3.1.Conv1Weight
    Layer3.1.Conv2Weight
    Layer4.0.Conv1Weight
    Layer4.0.Conv2Weight
    Layer4.0.Downsample.0Weight
    Layer4.1.Conv1Weight
    Layer4.1.Conv2Weight
    Fcweight"""
def frozen_mask(mask,model,itera):
    prunable_types=(nn.Conv2d, nn.Linear)
    if itera==1:
        for mod_name, mod in model.named_modules():
            if isinstance(mod, prunable_types):
                for name, param_layer in mod.named_parameters():
                    if ('bias' not in name) and (mod_name == 'conv1'):
                        prev_mask = torch.ones(param_layer.size(), dtype=torch.bool, requires_grad=False)
                        prev_mask = prev_mask.cuda()
                        mask[mod_name+name]=prev_mask
                    if ('bias' not in name) and ('layer1' in mod_name):
                        prev_mask = torch.ones(param_layer.size(), dtype=torch.bool, requires_grad=False)
                        prev_mask = prev_mask.cuda()
                        mask[mod_name+name]=prev_mask
                    """ if ('bias' not in name) and ('Fc' in mod_name):
                        prev_mask = torch.ones(param_layer.size(), dtype=torch.bool, requires_grad=False)
                        prev_mask = prev_mask.cuda()
                        mask[mod_name+name]=prev_mask """
    elif itera==2:
        for mod_name, mod in model.named_modules():
            if isinstance(mod, prunable_types):
                    for name, param_layer in mod.named_parameters():
                        if ('bias' not in name) and (mod_name == 'conv1'):
                            prev_mask = torch.ones(param_layer.size(), dtype=torch.bool, requires_grad=False)
                            prev_mask = prev_mask.cuda()
                            mask[mod_name+name]=prev_mask
                        if ('bias' not in name) and (('layer1' in mod_name)or('layer2' in mod_name)):
                            prev_mask = torch.ones(param_layer.size(), dtype=torch.bool, requires_grad=False)
                            prev_mask = prev_mask.cuda()
                            mask[mod_name+name]=prev_mask
                        """ if ('bias' not in name) and ('Fc' in mod_name):
                            prev_mask = torch.ones(param_layer.size(), dtype=torch.bool, requires_grad=False)
                            prev_mask = prev_mask.cuda()
                            mask[mod_name+name]=prev_mask """
    return mask

def zero_prun_gradient(model,mask):
    mask={key:mask[key].cuda() for key in mask}
    prunable_types=(nn.Conv2d, nn.Linear)
    for mod_name, mod in model.named_modules():
            if isinstance(mod,prunable_types):
                for name, param_layer in mod.named_parameters():
                    if 'bias' not in name and 'Fc' not in mod_name:
                        param_layer.grad = param_layer.grad * mask[mod_name+name]
    
    return model

def zero_fixed_gradient(model,pre_mask):
    prunable_types=(nn.Conv2d, nn.Linear)
    for mod_name, mod in model.named_modules():
        if isinstance(mod,prunable_types):
            for name, param_layer in mod.named_parameters():
                if 'bias' not in name and 'Fc' not in mod_name:
                    mask = torch.zeros(param_layer.size(), dtype=torch.bool, requires_grad=False)
                    mask = mask.cuda()
                    mask |= pre_mask[mod_name+name]
                    #print(param_layer.grad) #param_layer.grad一开始为None
                    #print(param_layer.is_cuda)
                    #print(mask.is_cuda)
                    #print(mod_name,name)
                    #print(type(param_layer.grad))
                    param_layer.grad = param_layer.grad * (~mask)
    return model

def get_all_prunable(model,pre_mask):
    prunable_types=(nn.Conv2d, nn.Linear)
    all_prunable = torch.tensor([]).cuda()
    for mod_name, mod in model.named_modules():
            if isinstance(mod,prunable_types):
                for name, param_layer in mod.named_parameters():
                    if 'bias' not in name:
                        prev_mask = torch.zeros(param_layer.size(), dtype=torch.bool, requires_grad=False)
                        prev_mask = prev_mask.cuda()

                        if mod_name+name in pre_mask:
                            #k=pre_mask[mod_name+name]
                            #print(k.is_cuda)
                            prev_mask |= pre_mask[mod_name+name]
                        p = param_layer.masked_select(~prev_mask)

                        if p is not None:
                            all_prunable = torch.cat((all_prunable.view(-1), p), -1)
    
    return all_prunable

def get_cutoff(all_prunable,prune_qu): 
    aprun_n1=all_prunable.cpu().detach().numpy()
    aprun_n=aprun_n1.copy()
    aprun_n=np.abs(aprun_n)

    cutoff=np.array(np.quantile(aprun_n, q=prune_qu))
    cutoff = cutoff.astype(np.float32)
    cutoff= torch.from_numpy(cutoff)
    cutoff=cutoff.cuda()
    #cutoff = torch.quantile(torch.abs(all_prunable), q=prune_qu)
    
    return cutoff

def get_all_fixable(model,pre_mask): 
    prunable_types=(nn.Conv2d, nn.Linear)
    all_fixable = torch.tensor([]).cuda()
    for mod_name, mod in model.named_modules():
            if isinstance(mod,prunable_types):
                for name, param_layer in mod.named_parameters():
                    if 'bias' not in name:
                        prev_mask = torch.zeros(param_layer.size(), dtype=torch.bool, requires_grad=False)
                        prev_mask = prev_mask.cuda()

                        if mod_name+name in pre_mask:
                            prev_mask |= pre_mask[mod_name+name]
                        p = param_layer.masked_select(~prev_mask)
                        if p is not None:
                            all_fixable = torch.cat((all_fixable.view(-1), p), -1)
    return all_fixable

"""Resnet34 structure
conv1.weight
bn1.weight
bn1.bias
layer1.0.conv1.weight
layer1.0.bn1.weight
layer1.0.bn1.bias
layer1.0.conv2.weight
layer1.0.bn2.weight
layer1.0.bn2.bias
layer1.1.conv1.weight
layer1.1.bn1.weight
layer1.1.bn1.bias
layer1.1.conv2.weight
layer1.1.bn2.weight
layer1.1.bn2.bias
layer1.2.conv1.weight
layer1.2.bn1.weight
layer1.2.bn1.bias
layer1.2.conv2.weight
layer1.2.bn2.weight
layer1.2.bn2.bias
layer2.0.conv1.weight
layer2.0.bn1.weight
layer2.0.bn1.bias
layer2.0.conv2.weight
layer2.0.bn2.weight
layer2.0.bn2.bias
layer2.0.downsample.0.weight
layer2.0.downsample.1.weight
layer2.0.downsample.1.bias
layer2.1.conv1.weight
layer2.1.bn1.weight
layer2.1.bn1.bias
layer2.1.conv2.weight
layer2.1.bn2.weight
layer2.1.bn2.bias
layer2.2.conv1.weight
layer2.2.bn1.weight
layer2.2.bn1.bias
layer2.2.conv2.weight
layer2.2.bn2.weight
layer2.2.bn2.bias
layer2.3.conv1.weight
layer2.3.bn1.weight
layer2.3.bn1.bias
layer2.3.conv2.weight
layer2.3.bn2.weight
layer2.3.bn2.bias
layer3.0.conv1.weight
layer3.0.bn1.weight
layer3.0.bn1.bias
layer3.0.conv2.weight
layer3.0.bn2.weight
layer3.0.bn2.bias
layer3.0.downsample.0.weight
layer3.0.downsample.1.weight
layer3.0.downsample.1.bias
layer3.1.conv1.weight
layer3.1.bn1.weight
layer3.1.bn1.bias
layer3.1.conv2.weight
layer3.1.bn2.weight
layer3.1.bn2.bias
layer3.2.conv1.weight
layer3.2.bn1.weight
layer3.2.bn1.bias
layer3.2.conv2.weight
layer3.2.bn2.weight
layer3.2.bn2.bias
layer3.3.conv1.weight
layer3.3.bn1.weight
layer3.3.bn1.bias
layer3.3.conv2.weight
layer3.3.bn2.weight
layer3.3.bn2.bias
layer3.4.conv1.weight
layer3.4.bn1.weight
layer3.4.bn1.bias
layer3.4.conv2.weight
layer3.4.bn2.weight
layer3.4.bn2.bias
layer3.5.conv1.weight
layer3.5.bn1.weight
layer3.5.bn1.bias
layer3.5.conv2.weight
layer3.5.bn2.weight
layer3.5.bn2.bias
layer4.0.conv1.weight
layer4.0.bn1.weight
layer4.0.bn1.bias
layer4.0.conv2.weight
layer4.0.bn2.weight
layer4.0.bn2.bias
layer4.0.downsample.0.weight
layer4.0.downsample.1.weight
layer4.0.downsample.1.bias
layer4.1.conv1.weight
layer4.1.bn1.weight
layer4.1.bn1.bias
layer4.1.conv2.weight
layer4.1.bn2.weight
layer4.1.bn2.bias
layer4.2.conv1.weight
layer4.2.bn1.weight
layer4.2.bn1.bias
layer4.2.conv2.weight
layer4.2.bn2.weight
layer4.2.bn2.bias
fc.weight
fc.bias """

def get_fixed(all_fixable,fixed_qu): 
    aprun_n1=all_fixable.cpu().detach().numpy()
    aprun_n=aprun_n1.copy()
    aprun_n=np.abs(aprun_n)
    fixed=np.array(np.quantile(aprun_n, q=(1-fixed_qu)))
    fixed=fixed.astype(np.float32)
    fixed=torch.from_numpy(fixed)
    fixed=fixed.cuda()   
    return fixed

def check_prune_instruct(prune_quantile):
    if not isinstance(prune_quantile, list):  
        assert 0 < prune_quantile < 1
        prune_quantile = [prune_quantile] * (2)
    assert len(prune_quantile) == 2, "Must give prune instructions for each task,except for the task 3"

def mask_prune(model,pre_mask,cutoff,fix):
    #pre_mask={key:pre_mask[key].cuda() for key in pre_mask}
    prunable_types=(nn.Conv2d, nn.Linear)
    mask = {}  
    with torch.no_grad():
        for mod_name, mod in model.named_modules():
            if isinstance(mod,prunable_types):
                for name, param_layer in mod.named_parameters():
                    if 'bias' not in name:
                        prev_mask = torch.zeros(param_layer.size(), dtype=torch.bool, requires_grad=False)  # p
                        prev_mask = prev_mask.cuda()
                        if mod_name+name in pre_mask:
                            prev_mask |= pre_mask[mod_name+name]
                        curr_mask = torch.abs(param_layer).ge(cutoff)  
                        curr_mask = torch.logical_and(curr_mask, ~prev_mask)  
                        curr_mask2 = torch.abs(param_layer).ge(fix)
                        curr_mask2 = torch.logical_or(curr_mask2, prev_mask)
                        param_layer *= (curr_mask | prev_mask)

                        mask[mod_name+name] = curr_mask2
    
    return model,mask

