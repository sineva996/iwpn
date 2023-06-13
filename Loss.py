import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

def cross_loss():
    loss_temp = torch.nn.CrossEntropyLoss()
    return loss_temp

def kl_loss():
    loss_temp = torch.nn.KLDivLoss()
    return loss_temp

def dist_loss(loss_soft,loss_hard,T,alpha):
    loss_soft2 = loss_soft *T *T
    loss_temp = (1-alpha)*loss_hard + alpha *loss_soft2
 
    return loss_temp

def get_allacc(target,predict):
    allacc=[0,0,0,0,0,0,0]
    c_counts = np.array([329,74,160,1185,478,162,680], dtype = 'float')
    for i in range(len(target)):
        if target[i]==predict[i]:
            k=target[i]
            allacc[k] += 1
    allacc = np.array(allacc, dtype = 'float')
    allacc /= c_counts
    mean=np.mean(allacc)
    std=np.std(allacc)
    allacc=np.append(allacc,mean)
    allacc=np.append(allacc,std)
    allacc = [float('{:.4f}'.format(i)) for i in allacc]
    return allacc

def get_allacc_af(target,predict):
    allacc=[0,0,0,0,0,0,0]
    c_counts = np.array([500,500,500,500,500,500,500], dtype = 'float')
    for i in range(len(target)):
        if target[i]==predict[i]:
            k=target[i]
            allacc[k] += 1
    allacc = np.array(allacc, dtype = 'float')
    allacc /= c_counts
    mean=np.mean(allacc)
    std=np.std(allacc)
    allacc=np.append(allacc,mean)
    allacc=np.append(allacc,std)
    allacc = [float('{:.4f}'.format(i)) for i in allacc]
    return allacc

def c_weight():
    c_counts=torch.from_numpy( np.array([1290,281,717,4772,1982,705,2524]).astype(np.float32))
    c_weight = (torch.sum(c_counts)-c_counts)/torch.sum(c_counts)
    return c_weight

def c_weight_af():
    c_counts=torch.from_numpy( np.array([14090,6378,3803,134415,25459,24882,74874]).astype(np.float32))
    c_weight = (torch.sum(c_counts)-c_counts)/torch.sum(c_counts)
    return c_weight

def c_weight_af1():
    c_counts=torch.from_numpy( np.array([14090,6378,3803,13441, 12729,12441,14975]).astype(np.float32))
    c_weight = (torch.sum(c_counts)-c_counts)/torch.sum(c_counts)
    return c_weight

class Affnity(nn.Module):
    def __init__(self, device, num_class=7, feat_dim=512):
        super(Affnity, self).__init__()
        self.num_class = num_class
        self.feat_dim = feat_dim
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.device = device

        self.centers = nn.Parameter(torch.randn(self.num_class, self.feat_dim).to(device))

    def forward(self, x, labels):
        x = self.gap(x).view(x.size(0), -1)

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_class) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_class, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_class).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_class)
        mask = labels.eq(classes.expand(batch_size, self.num_class))

        dist = distmat * mask.float()
        dist = dist / self.centers.var(dim=0).sum()

        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss