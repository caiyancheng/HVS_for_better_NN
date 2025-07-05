import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

transform_test = transforms.ToTensor()

class CIFAR100C(Dataset):
    def __init__(self, corruption_root, corruption_type='gaussian_noise', severity=1, transform=None):
        """
        corruption_root: path to the directory containing CIFAR-100-C
        corruption_type: one of the 15 corruptions, e.g., 'gaussian_noise'
        severity: 1-5
        """
        assert corruption_type in [
            'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
            'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness',
            'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
        ], "Invalid corruption type"

        assert 1 <= severity <= 5, "Severity should be in [1, 5]"

        self.data = np.load(os.path.join(corruption_root, f'{corruption_type}.npy'))
        self.targets = np.load(os.path.join(corruption_root, 'labels.npy'))

        # Each corruption has 5 severities, each with 10,000 samples
        start = (severity - 1) * 10000
        end = severity * 10000
        self.data = self.data[start:end]
        self.targets = self.targets[start:end]
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)

def dataset_load(dataset_name, type='train', corruption_type='gaussian_noise', severity=1):
    data_root = r'../Datasets/CIFAR10/data'
    if dataset_name == 'CIFAR-100':
        if type == 'train':
            trainset = torchvision.datasets.CIFAR100(root=data_root, train=True, download=False, transform=transform_train)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
            testset = torchvision.datasets.CIFAR100(root=data_root, train=False, download=False, transform=transform_test)
            testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
            return trainloader, testloader
        elif type == 'test':
            testset = torchvision.datasets.CIFAR100(root=data_root, train=False, download=False, transform=transform_test)
            testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
            return testloader

    elif dataset_name == 'CIFAR-10':
        if type == 'train':
            trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=False, transform=transform_train)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
            testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=False, transform=transform_test)
            testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
            return trainloader, testloader
        elif type == 'test':
            testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=False, transform=transform_test)
            testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
            return testloader

    elif dataset_name == 'CIFAR-100-C':
        corruption_root = r'../Datasets/CIFAR100-C'
        testset = CIFAR100C(corruption_root=corruption_root,
                            corruption_type=corruption_type,
                            severity=severity,
                            transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
        return testloader
