import numpy as np
import torch
import torchvision
# from color_space_transform import *
import torchvision.transforms as transforms

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

transform_test =  transforms.ToTensor()

def dataset_load(dataset_name):
    data_root = r'../Datasets/CIFAR10/data'
    if dataset_name == 'CIFAR-100':
        trainset = torchvision.datasets.CIFAR100(root=data_root, train=True, download=False, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
        testset = torchvision.datasets.CIFAR100(root=data_root, train=False, download=False, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
        return trainloader, testloader
    elif dataset_name == 'CIFAR-10':
        trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=False, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
        testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=False, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
        return trainloader, testloader