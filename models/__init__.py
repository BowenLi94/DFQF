from .generator import Generator
from .resnet_cifar import *
from .vgg import *
from .mobilenetv2 import *

import torch.nn as nn

def get_model(model_name,args):
    if 'resnet' in model_name:
        return resnet_cifar.__dict__[model_name](num_classes = args.num_classes)
    elif 'vgg' in model_name:
        return vgg.__dict__[model_name](num_classes = args.num_classes)
    elif 'mobilenetv2' in model_name:
        return mobilenetv2(num_classes = args.num_classes)
