#from mobilenetv2 import mobilenetv2
from resnet import resnet18
#from vgg16_bn_mine_cp import vgg16_bn_mine

import torch
import torchsummary

model = resnet18().to('cpu')
print(model.partition)
#data = torch.randn(1, 3, 224, 224).to('cpu')
#out = model(data)
#print(out)
#print(model)
#torchsummary.summary(model,(3,224,224),device='cpu')
#print(torch.cuda.is_available())
