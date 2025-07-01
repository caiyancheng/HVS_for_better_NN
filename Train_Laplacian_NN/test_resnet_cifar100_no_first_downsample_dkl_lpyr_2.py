import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import math
import os
from lpyr_dec import laplacian_pyramid_simple
from torchvision.models.resnet import BasicBlock
from torchvision.models import resnet18
import torch.nn.functional as F

# Viewing condition
peak_luminance = 100.0
checkpoint_path = '../HVS_for_better_NN_pth/best_resnet18_cifar100_no_first_downsample_dkl_lpyr_thin_pl100.0_2.pth'
resolution = [3840, 2160]
diagonal_size_inches = 55
viewing_distance_meters = 1

ar = resolution[0] / resolution[1]
height_mm = math.sqrt((diagonal_size_inches * 25.4)**2 / (1 + ar**2))
display_size_m = (ar * height_mm / 1000, height_mm / 1000)
pix_deg = 2 * math.degrees(math.atan(0.5 * display_size_m[0] / resolution[0] / viewing_distance_meters))
display_ppd = 1 / pix_deg

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
lpyr = laplacian_pyramid_simple(32, 32, display_ppd, device)

# Color space conversion
LMS2006_to_DKLd65 = torch.tensor([
    [1.0000, 1.0000, 0],
    [1.0000, -2.31113018, 0],
    [-1.0000, -1.0000, 50.97757133]
], dtype=torch.float32)

XYZ_to_LMS2006 = torch.tensor([
    [0.18759627, 0.58516865, -0.02638426],
    [-0.13339743, 0.40550578, 0.03450213],
    [0.00024438, -0.00054300, 0.01940685]
], dtype=torch.float32)

def srgb_to_linear_rgb(srgb):
    threshold = 0.04045
    return torch.where(
        srgb <= threshold,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4
    )

def linear_rgb_to_xyz(rgb):
    M = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                      [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]], dtype=rgb.dtype, device=rgb.device)
    rgb = rgb.permute(1, 2, 0)
    xyz = torch.tensordot(rgb, M.T, dims=1)
    return xyz.permute(2, 0, 1)

def xyz_to_lms2006(xyz):
    xyz = xyz.permute(1, 2, 0)
    lms = torch.tensordot(xyz, XYZ_to_LMS2006.T.to(xyz.device), dims=1)
    return lms.permute(2, 0, 1)

def lms_to_dkl(lms):
    lms = lms.permute(1, 2, 0)
    dkl = torch.tensordot(lms, LMS2006_to_DKLd65.T.to(lms.device), dims=1)
    return dkl.permute(2, 0, 1)

class RGBtoDKLTransform:
    def __init__(self, peak_luminance=500.0):
        self.peak_luminance = peak_luminance

    def __call__(self, tensor):
        tensor = srgb_to_linear_rgb(tensor)
        tensor = linear_rgb_to_xyz(tensor) * self.peak_luminance
        tensor = xyz_to_lms2006(tensor)
        tensor = lms_to_dkl(tensor)
        return tensor

# CIFAR100 Test Set
data_root = '../Datasets/CIFAR10/data'
transform_test = transforms.Compose([
    transforms.ToTensor(),
    RGBtoDKLTransform(peak_luminance=peak_luminance)
])

testset = torchvision.datasets.CIFAR100(root=data_root, train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

# Model definition
def make_layer(block, in_planes, out_planes, blocks, stride=1):
    downsample = None
    if stride != 1 or in_planes != out_planes:
        downsample = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_planes),
        )
    layers = [block(in_planes, out_planes, stride, downsample)]
    for _ in range(1, blocks):
        layers.append(block(out_planes, out_planes))
    return nn.Sequential(*layers)

class PyramidResNet18(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        base = resnet18(weights=None)
        self.channel = [64, 64, 128, 256, 512]
        base.conv1 = nn.Conv2d(3, self.channel[0], kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()

        self.conv1 = base.conv1
        self.bn1 = nn.BatchNorm2d(self.channel[0])
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = make_layer(BasicBlock, self.channel[0], self.channel[1], 2, 1)
        self.layer2 = make_layer(BasicBlock, self.channel[1], self.channel[2], 2, 2)
        self.layer3 = make_layer(BasicBlock, self.channel[2], self.channel[3], 2, 2)
        self.layer4 = make_layer(BasicBlock, self.channel[3], self.channel[4], 2, 2)
        self.avgpool = base.avgpool
        self.fc = nn.Linear(self.channel[4], num_classes)

    def forward(self, x):
        _, pyr = lpyr.decompose(x, levels=4)
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return self.fc(torch.flatten(x, 1))

# Load model
model = PyramidResNet18(num_classes=100).to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint)
model.eval()

# Run test
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

print(f"✅ Test Accuracy: {100 * correct / total:.2f}%")
