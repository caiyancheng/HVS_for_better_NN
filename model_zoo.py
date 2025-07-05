import torch
from torchvision.models import resnet18
import torch.nn as nn

def model_create(model_name, dataset_name):
    if model_name == 'resnet18' and (dataset_name == 'CIFAR-100' or dataset_name == 'CIFAR-10') :
        model = resnet18(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        if dataset_name == 'CIFAR-100':
            model.fc = nn.Linear(model.fc.in_features, 100)
        elif dataset_name == 'CIFAR-10':
            model.fc = nn.Linear(model.fc.in_features, 10)
        return model
    else:
        raise NotImplementedError('The setting is not implemented.')