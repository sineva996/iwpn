from torch import nn
from torch.nn import functional as F
import torch
import torch.nn.init as init
from torchvision import models


def resconv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
                     
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = resconv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = resconv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, pretrained=False, num_classes=7, drop_rate=0):
        super(ResNet18, self).__init__()
        self.drop_rate = drop_rate
        resnet = models.resnet18(pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.fc = nn.Linear(512, num_classes)


    def forward(self, x):
        x = self.features(x)
        if self.drop_rate > 0:
            x =  nn.Dropout(self.drop_rate)(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)

        return out, out

""" IWPN_net_struction
conv1.weight
bn1.weight
bn1.bias
bn1.running_mean
bn1.running_var
bn1.num_batches_tracked
layer1.0.conv1.weight
layer1.0.bn1.weight
layer1.0.bn1.bias
layer1.0.bn1.running_mean
layer1.0.bn1.running_var
layer1.0.bn1.num_batches_tracked
layer1.0.conv2.weight
layer1.0.bn2.weight
layer1.0.bn2.bias
layer1.0.bn2.running_mean
layer1.0.bn2.running_var
layer1.0.bn2.num_batches_tracked
layer1.1.conv1.weight
layer1.1.bn1.weight
layer1.1.bn1.bias
layer1.1.bn1.running_mean
layer1.1.bn1.running_var
layer1.1.bn1.num_batches_tracked
layer1.1.conv2.weight
layer1.1.bn2.weight
layer1.1.bn2.bias
layer1.1.bn2.running_mean
layer1.1.bn2.running_var
layer1.1.bn2.num_batches_tracked
layer2.0.conv1.weight
layer2.0.bn1.weight
layer2.0.bn1.bias
layer2.0.bn1.running_mean
layer2.0.bn1.running_var
layer2.0.bn1.num_batches_tracked
layer2.0.conv2.weight
layer2.0.bn2.weight
layer2.0.bn2.bias
layer2.0.bn2.running_mean
layer2.0.bn2.running_var
layer2.0.bn2.num_batches_tracked
layer2.0.downsample.0.weight
layer2.0.downsample.1.weight
layer2.0.downsample.1.bias
layer2.0.downsample.1.running_mean
layer2.0.downsample.1.running_var
layer2.0.downsample.1.num_batches_tracked
layer2.1.conv1.weight
layer2.1.bn1.weight
layer2.1.bn1.bias
layer2.1.bn1.running_mean
layer2.1.bn1.running_var
layer2.1.bn1.num_batches_tracked
layer2.1.conv2.weight
layer2.1.bn2.weight
layer2.1.bn2.bias
layer2.1.bn2.running_mean
layer2.1.bn2.running_var
layer2.1.bn2.num_batches_tracked
layer3.0.conv1.weight
layer3.0.bn1.weight
layer3.0.bn1.bias
layer3.0.bn1.running_mean
layer3.0.bn1.running_var
layer3.0.bn1.num_batches_tracked
layer3.0.conv2.weight
layer3.0.bn2.weight
layer3.0.bn2.bias
layer3.0.bn2.running_mean
layer3.0.bn2.running_var
layer3.0.bn2.num_batches_tracked
layer3.0.downsample.0.weight
layer3.0.downsample.1.weight
layer3.0.downsample.1.bias
layer3.0.downsample.1.running_mean
layer3.0.downsample.1.running_var
layer3.0.downsample.1.num_batches_tracked
layer3.1.conv1.weight
layer3.1.bn1.weight
layer3.1.bn1.bias
layer3.1.bn1.running_mean
layer3.1.bn1.running_var
layer3.1.bn1.num_batches_tracked
layer3.1.conv2.weight
layer3.1.bn2.weight
layer3.1.bn2.bias
layer3.1.bn2.running_mean
layer3.1.bn2.running_var
layer3.1.bn2.num_batches_tracked
layer4.0.conv1.weight
layer4.0.bn1.weight
layer4.0.bn1.bias
layer4.0.bn1.running_mean
layer4.0.bn1.running_var
layer4.0.bn1.num_batches_tracked
layer4.0.conv2.weight
layer4.0.bn2.weight
layer4.0.bn2.bias
layer4.0.bn2.running_mean
layer4.0.bn2.running_var
layer4.0.bn2.num_batches_tracked
layer4.0.downsample.0.weight
layer4.0.downsample.1.weight
layer4.0.downsample.1.bias
layer4.0.downsample.1.running_mean
layer4.0.downsample.1.running_var
layer4.0.downsample.1.num_batches_tracked
layer4.1.conv1.weight
layer4.1.bn1.weight
layer4.1.bn1.bias
layer4.1.bn1.running_mean
layer4.1.bn1.running_var
layer4.1.bn1.num_batches_tracked
layer4.1.conv2.weight
layer4.1.bn2.weight
layer4.1.bn2.bias
layer4.1.bn2.running_mean
layer4.1.bn2.running_var
layer4.1.bn2.num_batches_tracked
fc.weight
fc.bias
bn.weight
bn.bias
bn.running_mean
bn.running_var
bn.num_batches_tracked """
class ResNet18_IWPN(nn.Module):
    def __init__(self, num_class=7):
        super(ResNet18_IWPN, self).__init__()
        resnet = models.resnet18(True)
        checkpoint = torch.load('./net/pre-models/resnet18_msceleb.pth')
        resnet.load_state_dict(checkpoint['state_dict'],strict=True)
        
        self.conv1=resnet.conv1
        self.bn1=resnet.bn1
        #self.layer = nn.Sequential(*list(resnet.children())[:-2])
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.sig = nn.Sigmoid()
        self.fc = nn.Linear(100352, num_class)
        self.bn = nn.BatchNorm1d(num_class)

    def forward(self, x):
        #print(x.shape)
        #x=x.unsqueeze(2).unsqueeze(3)
        #x = self.feat_net(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        out = x.view(x.size(0), -1)
        out = self.fc(out)
        out = self.bn(out)
   
        return out, x

class ResNet34_IWPN(nn.Module):
    def __init__(self, num_class=7):
        super(ResNet34_IWPN, self).__init__()
        resnet = models.resnet34(pretrained=True,progress=False)
        numFit = resnet.fc.in_features
        
        self.conv1=resnet.conv1
        self.bn1=resnet.bn1
        #self.layer = nn.Sequential(*list(resnet.children())[:-2])
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.sig = nn.Sigmoid()
        self.fc=nn.Sequential(
            nn.Linear(100352, 512),
            nn.Linear(512, num_class)
            )
        
        self.bn = nn.BatchNorm1d(num_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        out = x.view(x.size(0), -1)
        out = self.fc(out)
        out = self.bn(out)
   
        return out, x

