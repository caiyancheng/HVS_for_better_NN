import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
import math
from lpyr_dec import laplacian_pyramid_simple

# ==== 参数设置 ====
peak_luminance = 500.0
checkpoint_path = f'../HVS_for_better_NN_pth/best_resnet18_cifar10_no_first_downsample_dkl_lpyr_pl{peak_luminance}_5.pth'
resolution = [3840, 2160]
diagonal_size_inches = 55
viewing_distance_meters = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==== PPD 计算 ====
ar = resolution[0] / resolution[1]
height_mm = math.sqrt((diagonal_size_inches * 25.4) ** 2 / (1 + ar ** 2))
display_size_m = (ar * height_mm / 1000, height_mm / 1000)
pix_deg = 2 * math.degrees(math.atan(0.5 * display_size_m[0] / resolution[0] / viewing_distance_meters))
display_ppd = 1 / pix_deg
lpyr = laplacian_pyramid_simple(32, 32, display_ppd, device)

# ==== 色彩空间转换 ====
LMS2006_to_DKLd65 = torch.tensor([
    [1.0, 1.0, 0],
    [1.0, -2.311130179947035, 0],
    [-1.0, -1.0, 50.977571328718781]
], dtype=torch.float32)
XYZ_to_LMS2006 = torch.tensor([
    [0.18759627, 0.58516865, -0.02638426],
    [-0.13339743, 0.40550578, 0.03450213],
    [0.00024438, -0.000543, 0.01940685]
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

# ==== 加载数据 ====
data_root = '../Datasets/CIFAR10/data'
transform_test = transforms.Compose([
    transforms.ToTensor(),
    RGBtoDKLTransform(peak_luminance=peak_luminance)
])
testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

# ==== 模型定义 ====
from torchvision.models import resnet18

class PyramidResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        base = resnet18(weights=None)
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = nn.Linear(base.fc.in_features, num_classes)

        self.inject1 = nn.Conv2d(3, 64, 1)
        self.inject2 = nn.Conv2d(3, 64, 1)
        self.inject3 = nn.Conv2d(3, 128, 1)
        self.inject4 = nn.Conv2d(3, 256, 1)

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )

    def forward(self, x, pyr):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        feat1 = self.inject1(F.interpolate(pyr[1], size=x.shape[-2:]))
        x = self.layer1(x * self.gate(feat1))
        feat2 = self.inject2(F.interpolate(pyr[2], size=x.shape[-2:]))
        x = self.layer2(x * self.gate(feat2))
        feat3 = self.inject3(F.interpolate(pyr[3], size=x.shape[-2:]))
        x = self.layer3(x * self.gate(feat3))
        feat4 = self.inject4(F.interpolate(pyr[4], size=x.shape[-2:]))
        x = self.layer4(x * self.gate(feat4))
        x = self.avgpool(x)
        return self.fc(torch.flatten(x, 1))

# ==== 加载模型 ====
model = PyramidResNet18().to(device)
assert os.path.isfile(checkpoint_path), f"❌ Checkpoint not found: {checkpoint_path}"
print(f"✅ Loading weights from {checkpoint_path}")
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# ==== 评估 ====
def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            gpyr_results, lpyr_results = lpyr.decompose(inputs)
            outputs = model(inputs, lpyr_results)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100.0 * correct / total
    print(f"🎯 Test Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    test()
