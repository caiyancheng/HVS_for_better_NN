import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from PIL import Image
import os
from torchvision.datasets import ImageFolder
from collections import defaultdict
from torchvision import datasets


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

def dataset_load(dataset_name, batch_size=128, type='train', corruption_type='gaussian_noise', severity=1, num_classes=None):
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    transform_test = transforms.ToTensor()

    data_root = r'../Datasets/CIFAR10/data'
    if dataset_name == 'CIFAR-100':
        if type == 'train':
            trainset = torchvision.datasets.CIFAR100(root=data_root, train=True, download=False, transform=transform_train)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
            return trainloader
        elif type == 'test':
            testset = torchvision.datasets.CIFAR100(root=data_root, train=False, download=False, transform=transform_test)
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
            return testloader

    elif dataset_name == 'CIFAR-10':
        if type == 'train':
            trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=False, transform=transform_train)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
            return trainloader
        elif type == 'test':
            testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=False, transform=transform_test)
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
            return testloader

    elif dataset_name == 'CIFAR-100-C' and type == 'test':
        corruption_root = r'../Datasets/CIFAR100-C'
        testset = CIFAR100C(corruption_root=corruption_root,
                            corruption_type=corruption_type,
                            severity=severity,
                            transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
        return testloader

    elif dataset_name == 'Tiny-ImageNet-C' and type == 'test':
        corruption_root = '../Datasets/Tiny-ImageNet-C'
        # corruption_root = 'E:\Datasets\Tiny-ImageNet-C\Tiny-ImageNet-C/'
        corruption_dir = os.path.join(corruption_root, corruption_type, str(severity))
        if not os.path.exists(corruption_dir):
            raise FileNotFoundError(f"Directory {corruption_dir} not found. Please check corruption type and path.")
        dataset = datasets.ImageFolder(corruption_dir, transform=transform_test)
        if num_classes is not None:
            class_indices = {cls: idx for idx, cls in enumerate(sorted(dataset.classes))}
            selected_classes = set(list(class_indices.keys())[:num_classes])
            selected_idx = [i for i, (img, label) in enumerate(dataset.samples)
                            if dataset.classes[label] in selected_classes]
            dataset = Subset(dataset, selected_idx)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        return dataloader

    elif dataset_name == 'Tiny-ImageNet':
        tiny_root = '../Datasets/tiny-imagenet-200'
        # tiny_root = r'E:\Datasets\tiny-imagenet-200\tiny-imagenet-200'
        # tiny_root = r'E:\Datasets\tiny-imagenet-200\tiny-imagenet-200'
        split = 'train' if type == 'train' else 'val'
        # Tiny-ImageNet has a special structure, need to restructure val set
        if split == 'val':
            val_dir = os.path.join(tiny_root, 'val')
            val_img_dir = os.path.join(val_dir, 'images')
            val_annotations_file = os.path.join(val_dir, 'val_annotations.txt')
            # Create class-based subfolders if not already done
            class_to_files = defaultdict(list)
            with open(val_annotations_file, 'r') as f:
                for line in f:
                    filename, classname, *_ = line.strip().split('\t')
                    class_to_files[classname].append(filename)
            for classname, files in class_to_files.items():
                class_dir = os.path.join(val_img_dir, classname)
                os.makedirs(class_dir, exist_ok=True)
                for fname in files:
                    src = os.path.join(val_img_dir, fname)
                    dst = os.path.join(class_dir, fname)
                    if not os.path.exists(dst):
                        os.link(src, dst)
            data_dir = val_img_dir  # reorganized val folder
        else:
            data_dir = os.path.join(tiny_root, 'train')
        dataset = datasets.ImageFolder(data_dir, transform=transform_train if type == 'train' else transform_test)
        # Restrict to first `num_classes` if specified
        if num_classes is not None:
            class_indices = {cls: idx for idx, cls in enumerate(sorted(dataset.classes))}
            selected_classes = set(list(class_indices.keys())[:num_classes])
            selected_idx = [i for i, (img, label) in enumerate(dataset.samples) if
                            dataset.classes[label] in selected_classes]
            dataset = Subset(dataset, selected_idx)
        dataloader = DataLoader(dataset, batch_size=batch_size if type == 'train' else 100, shuffle=(type == 'train'),
                                num_workers=4)
        return dataloader
    else:
        raise ValueError('Invalid dataset_name')