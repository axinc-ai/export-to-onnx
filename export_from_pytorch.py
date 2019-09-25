import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms

vgg16 = models.vgg16(pretrained=True)

x = Variable(torch.randn(1, 3, 224, 224))
torch.onnx.export(vgg16, x, 'vgg16_pytorch.onnx', verbose=True, opset_version=10)
