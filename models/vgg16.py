'''
Modified from https://github.com/pytorch/vision.git
'''
import math
from custom import ChannelPruning
import torch.nn as nn
import torch.nn.init as init

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class VGG_mine(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, cfg):
        super(VGG_mine, self).__init__()
        layer1,layer2,layer3,layer4,layer5 = make_layers(cfg, batch_norm=True)
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4
        self.layer5 = layer5
        self.pruning = False
        self.channel_prune1 = ChannelPruning(64,64)
        self.channel_prune2 = ChannelPruning(128,128)
        self.channel_prune3 = ChannelPruning(256,256)
        self.channel_prune4 = ChannelPruning(512,512)
        #self.channel_prune5 = ChannelPruning(512,512)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.layer1(x)
        print(x.shape)
        if self.pruning is not False and self.channel_pruning.rate !=0 :
            if self.partition==1.0:
                x = x * self.channel_prune1(out)
        print(x.shape)
        x = self.layer2(x)
        print(x.shape)
        if self.pruning is not False and self.channel_pruning.rate !=0 :
            if self.partition==2.0:
                x = x * self.channel_prune2(out)
        print(x.shape)
        x = self.layer3(x)
        print(x.shape)
        if self.pruning is not False and self.channel_pruning.rate !=0 :
            if self.partition==3.0:
                x = x * self.channel_prune3(out)
        print(x.shape)
        x = self.layer4(x)
        print(x.shape)
        if self.pruning is not False and self.channel_pruning.rate !=0 :
            if self.partition==4.0:
                x = x * self.channel_prune4(out)
        print(x.shape)
        x = self.layer5(x)
        print(x.size)
        x = x.view(-1,x.size(1))
        print(x.shape)
        x = self.classifier(x)
        return x
"""
def make_layers_mine(cfg, batch_norm=False):
    layers = []
    in_channels = 3
	i_ = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
"""

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    i_ = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            globals()['layer'+str(i_)]=layers
            layers=[]
            i_+=1
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layer1),nn.Sequential(*layer2),nn.Sequential(*layer3),nn.Sequential(*layer4),nn.Sequential(*layer5) 


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))


def vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']))


def vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True))

def vgg16_bn_mine():
	return VGG_mine(cfg['D'])


def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']))


def vgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True))
