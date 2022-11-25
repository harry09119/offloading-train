# -*- coding=utf-8 -*-

__all__ = [
    'ResNet',
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152',
    'resnext50_32x4d',
    'resnext101_32x8d',
    'wide_resnet50_2',
    'wide_resnet101_2',
]
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Optional, Callable, List
from . import ChannelPruning, SwitchableBatchNorm2d
import matplotlib.pyplot as plt

global partition_point
partition_point = False

def conv3x3(in_planes: int,
            out_planes: int,
            stride: int = 1,
            groups: int = 1,
            dilation: int = 1,
            normalization: str = '',
            activation: str = '') -> nn.Module:
    layers = []
    layers.append(
        nn.Conv2d(in_planes,
                  out_planes,
                  kernel_size=3,
                  stride=stride,
                  groups=groups,
                  padding=dilation,
                  dilation=dilation,
                  bias=False)
    )
    if normalization == 'bn':
        layers.append(nn.BatchNorm2d(out_planes))
        #layers.append(SwitchableBatchNorm2d(out_planes))
    if activation == 'relu':
        layers.append(nn.ReLU())
    #return ChannelPruning(*layers)
    return nn.Sequential(*layers)

def conv1x1(in_planes: int,
            out_planes: int,
            stride: int = 1,
            normalization: str = '',
            activation: str = '') -> nn.Module:
    layers = []
    layers.append(
        nn.Conv2d(in_planes,
                  out_planes,
                  kernel_size=1,
                  stride=stride,
                  bias=False)
    )
    if normalization == 'bn':
        layers.append(nn.BatchNorm2d(out_planes))
        #layers.append(SwitchableBatchNorm2d(out_planes))
    if activation == 'relu':
        layers.append(nn.ReLU())
    #return ChannelPruning(*layers)
    return nn.Sequential(*layers)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Optional[Callable] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 pruning: Optional[Callable] = None,
                 dilation: int = 1) -> None:
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise ValueError('Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride, normalization='bn', activation='relu')
        self.conv2 = conv3x3(planes, planes, normalization='bn')
        self.downsample = downsample
        self.stride = stride
        self.pruning = pruning
        self.channel_pruning = ChannelPruning(inplanes, planes)
        self.out_channels = planes
        self.in_channels = inplanes
        #self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global partition_point
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
                identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        if partition_point == True and self.pruning is not None and self.channel_pruning.rate != 0:
        #if partition_point == True and self.pruning is not None:
            #print("nonzero before pruning:", torch.count_nonzero(out))
            #print("nonzero after pruning:", torch.count_nonzero(out*self.channel_pruning(out)))
            #print("channel_pruning :", torch.sort(self.channel_pruning(out), dim=1, descending=True ))
            #sorted, indices = torch.sort(self.channel_pruning(out), dim=1, descending=True)
            #print("sorted :", sorted)
            #pruning_result = torch.stack([torch.index_select(out[i], 0, indices[i].squeeze()) for i in range(out.size(0))]) * sorted
            #print("set pruning_rate :", self.channel_pruning.rate)
            #print("out :", out)
            #print("channel value :", pruning_result)
            #return pruning_result
            #torch.sort(pruning_result, dim=1, descending=True)

            #channel slicing
            #if self.channel_pruning.rate < 0.9:
                #slicing_num = int(self.out_channels * self.channel_pruning.rate)
                #out = 
            #else:
            return out * self.channel_pruning(out)
            
            #return out * sorted
        else:
            return out
            
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Optional[Callable] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1) -> None:
        super().__init__()
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width, normalization='bn', activation='relu')
        self.conv2 = conv3x3(width, width, stride, groups, dilation, normalization='bn', activation='relu')
        self.conv3 = conv1x1(width, planes * self.expansion, normalization='bn')
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = F.relu(out)
        return out

class yolo_layers(nn.Module):
    def __init__(self,in_channels=512,num_classes=20):
        super(yolo_layers, self).__init__()
        self.num_classes = 20
        self.layer20 = nn.Sequential(
            nn.Conv2d(in_channels, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024, momentum=0.01),
            nn.LeakyReLU())
        self.layer21 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024, momentum=0.01),
            nn.LeakyReLU())
        self.layer22 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024, momentum=0.01),
            nn.LeakyReLU())

        # LAYER 6
        
        self.layer23 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024, momentum=0.01),
            nn.LeakyReLU())
        self.layer24 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024, momentum=0.01),
            nn.LeakyReLU())

        self.fc1 = nn.Sequential(
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(),
            #nn.Dropout(self.dropout_prop)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(4096, 7 * 7 * ((5) + self.num_classes))
        )

    def forward(self,x):
        #print("0:",x.shape)
        out = self.layer20(x)
        #print("1:",out.shape)
        out = self.layer21(out)
        #print("2:",out.shape)
        out = self.layer22(out)
        #print("3:",out.shape)
        out = self.layer23(out)
        #print("4:",out.shape)
        out = self.layer24(out)
        #print("5:",out.shape)
        out = out.reshape(out.size(0), -1)
        #print("6:",out.shape)
        out = self.fc1(out)
        #print("7:",out.shape)
        out = self.fc2(out)
        #print("8:",out.shape)
        out = torch.reshape(out,(-1,7,7,((5) + self.num_classes)))
        #print((5) + self.num_classes)
        #print("9:",out.shape)
        out[:, :, :, 0] = torch.sigmoid(out[:, :, :, 0])  # sigmoid to objness1_output
        out[:, :, :, 5:] = torch.sigmoid(out[:, :, :, 5:])  # sigmoid to class_output

        return out


class detnet_bottleneck(nn.Module):
    # no expansion
    # dilation = 2
    # type B use 1x1 conv
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, block_type='A'):
        super(detnet_bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=2, bias=False,dilation=2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes or block_type=='B':
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self,
                 block: Union[BasicBlock, Bottleneck],
                 yolo_block : yolo_layers,
                 layers: List[int],
                 num_classes: int = 1000,
                 zero_init_residual: bool = False,
                 groups: int = 1,
                 width_per_group: int = 64,
                 replace_stride_with_dilation=None) -> None:
        super().__init__()
        
        self.partition = 1
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None or 3-element tuple')
        self.groups = groups
        self.base_width = width_per_group
        self.conv1=nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        #self.layer5 = self._make_detnet_layer(in_channels=512)
        self.yolos = [yolo_block(512,num_classes)]
        self.layer5 = nn.Sequential(*self.yolos)
        
        #self.conv_end = nn.Conv2d(256, 30, kernel_size=3, stride=1, padding=1, bias=False)
        #self.bn_end = nn.BatchNorm2d(30)
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.conv3.norm.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.conv2.norm.weight, 0)
        """ 

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False) -> nn.Sequential:
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = conv1x1(self.inplanes, planes * block.expansion, stride, normalization='bn')
        layers = []
        layers.append(block(self.inplanes,
                            planes,
                            stride=stride,
                            downsample=downsample,
                            groups=self.groups,
                            base_width=self.base_width,
                            dilation=previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            if _ == blocks-1:
                layers.append(block(self.inplanes,
                                    planes,
                                    groups=self.groups,
                                    base_width=self.base_width,
                                    dilation=self.dilation,
                                    pruning=2))    
            else:
                layers.append(block(self.inplanes,
                                    planes,
                                    groups=self.groups,
                                    base_width=self.base_width,
                                    dilation=self.dilation))    

            #layers.append(block(self.inplanes,
                                #planes,
                                #groups=self.groups,
                                #base_width=self.base_width,
                                #dilation=self.dilation))
        return nn.Sequential(*layers)

    def _make_detnet_layer(self, in_channels):
        layers = []
        layers.append(detnet_bottleneck(in_planes=in_channels, planes=1024, block_type='B'))
        layers.append(detnet_bottleneck(in_planes=1024, planes=1024, block_type='A'))
        layers.append(detnet_bottleneck(in_planes=1024, planes=1024, block_type='A'))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global partition_point
        x = self.conv1(x)
        x = self.maxpool(x)
        
        if self.partition == 1.0:
            partition_point=True
        
        x = self.layer1(x)
        #print("-4:",x.shape)
        if self.partition == 1.0:
            partition_point=False
        elif self.partition == 2.0:
            partition_point=True
        
        x = self.layer2(x)
        #print("-3:",x.shape)
        if self.partition == 2.0:
            partition_point=False
        elif self.partition == 3.0:
            partition_point=True
        
        x = self.layer3(x)
        #print("-2:",x.shape)
        if self.partition == 3.0:
            partition_point=False
        elif self.partition == 4.0:
            partition_point = True

        x = self.layer4(x)

        x = self.layer5(x)
        return x
    
def _resnet(block, yolo_block,layers, **kwargs):
    return ResNet(block, yolo_block,layers, **kwargs)

def resnet18(**kwargs):
    return _resnet(BasicBlock, yolo_layers ,[2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return _resnet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs):
    return _resnet(Bottleneck, [3, 8, 36, 3], **kwargs)


def resnext50_32x4d(**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnext101_32x8d(**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


def wide_resnet50_2(**kwargs):
    kwargs['width_per_group'] = 128
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def wide_resnet101_2(**kwargs):
    kwargs['width_per_group'] = 128
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)
