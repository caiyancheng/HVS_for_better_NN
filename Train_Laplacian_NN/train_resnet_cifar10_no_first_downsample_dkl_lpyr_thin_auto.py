import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
import os
from tqdm import tqdm
from torchsummary import summary
import math
from lpyr_dec import *
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock
import copy
import json

# Viewing Condition Setting
peak_luminance = 500.0
checkpoint_path = f'../HVS_for_better_NN_pth/best_resnet18_cifar10_no_first_downsample_dkl_lpyr_thin_pl{peak_luminance}_3.pth'
load_pretrained_weights = False
resolution = [3840,2160]
diagonal_size_inches = 55
viewing_distance_meters = 1

ar = resolution[0]/resolution[1]
height_mm = math.sqrt( (diagonal_size_inches*25.4)**2 / (1+ar**2) )
display_size_m = (ar*height_mm/1000, height_mm/1000)
pix_deg = 2 * math.degrees(math.atan(0.5 * display_size_m[0] / resolution[0] / viewing_distance_meters))
display_ppd = 1 / pix_deg

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
lpyr = laplacian_pyramid_simple(32, 32, display_ppd, device)
target_acc = 93.9
save_log = {}


# --- ✅ DKL转换相关矩阵 ---
LMS2006_to_DKLd65 = torch.tensor([
  [1.000000000000000,   1.000000000000000,                   0],
  [1.000000000000000,  -2.311130179947035,                   0],
  [-1.000000000000000,  -1.000000000000000,  50.977571328718781]
], dtype=torch.float32)
XYZ_to_LMS2006 = torch.tensor([
   [0.187596268556126,   0.585168649077728,  -0.026384263306304],
   [-0.133397430663221,   0.405505777260049,   0.034502127690364],
   [0.000244379021663,  -0.000542995890619,   0.019406849066323]
], dtype=torch.float32)

# --- ✅ sRGB → Linear RGB ---
def srgb_to_linear_rgb(srgb):
    """sRGB 转线性 RGB (gamma 解码)，输入 [0, 1]，输出 [0, 1]"""
    threshold = 0.04045
    linear = torch.where(
        srgb <= threshold,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4
    )
    return linear

# --- ✅ Linear RGB → XYZ (D65) ---
def linear_rgb_to_xyz(rgb):
    """线性 RGB → XYZ，D65"""
    # 使用 sRGB 的 D65 到 XYZ 转换矩阵
    M = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                      [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]], dtype=rgb.dtype, device=rgb.device)
    rgb = rgb.permute(1, 2, 0)  # (C,H,W) → (H,W,C)
    xyz = torch.tensordot(rgb, M.T, dims=1)  # (H,W,3)
    return xyz.permute(2, 0, 1)  # (3,H,W)

# --- ✅ XYZ → LMS2006 ---
def xyz_to_lms2006(xyz):
    xyz = xyz.permute(1, 2, 0)  # C,H,W → H,W,C
    lms = torch.tensordot(xyz, XYZ_to_LMS2006.T.to(xyz.device), dims=1)
    return lms.permute(2, 0, 1)

# --- ✅ LMS2006 → DKL ---
def lms_to_dkl(lms):
    lms = lms.permute(1, 2, 0)  # C,H,W → H,W,C
    dkl = torch.tensordot(lms, LMS2006_to_DKLd65.T.to(lms.device), dims=1)
    return dkl.permute(2, 0, 1)  # C,H,W

# --- ✅ 最终 transform（sRGB → DKL） ---
class RGBtoDKLTransform:
    def __init__(self, peak_luminance=500.0):
        self.peak_luminance = peak_luminance

    def __call__(self, tensor):
        tensor = srgb_to_linear_rgb(tensor)               # sRGB → linear RGB
        tensor = linear_rgb_to_xyz(tensor) * self.peak_luminance  # → XYZ (cd/m²)
        tensor = xyz_to_lms2006(tensor)                   # → LMS
        tensor = lms_to_dkl(tensor)                       # → DKL
        return tensor

# 替换为原始字符串避免 warning
# data_root = r'E:\Datasets\CIFAR10\data'
data_root = r'../Datasets/CIFAR10/data'

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    RGBtoDKLTransform(peak_luminance=peak_luminance)  # ✅ DKL 替换 XYZ
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    RGBtoDKLTransform(peak_luminance=peak_luminance)
])

trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

def make_layer(block, in_planes, out_planes, blocks, stride=1):
    """自定义layer构造函数以兼容不同通道数"""
    downsample = None
    if stride != 1 or in_planes != out_planes:
        downsample = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_planes),
        )

    layers = []
    layers.append(block(in_planes, out_planes, stride, downsample))
    for _ in range(1, blocks):
        layers.append(block(out_planes, out_planes))

    return nn.Sequential(*layers)

class PyramidResNet18(nn.Module):
    def __init__(self, num_classes=10, channel_net=[4, 64, 128, 256, 512]):
        super().__init__()
        base = resnet18(weights=None)
        self.channel = channel_net #最初是[64, 64, 128, 256, 512] #64-93.90%, 32-94.03%, 16-94.39%， 8-94.16%, 4-94.04%
        # base.conv1 = nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 输入 concat image + L0
        base.conv1 = nn.Conv2d(3, self.channel[0], kernel_size=3, stride=1, padding=1, bias=False)  # 输入 concat image + L0
        base.maxpool = nn.Identity()

        self.conv1 = base.conv1
        self.bn1 = nn.BatchNorm2d(self.channel[0]) ###卧槽！反而+0.4%的正向增长
        self.relu = base.relu
        self.maxpool = base.maxpool
        # self.layer1 = base.layer1

        self.layer1 = make_layer(BasicBlock, self.channel[0], self.channel[1], blocks=2, stride=1)
        self.layer2 = make_layer(BasicBlock, self.channel[1], self.channel[2], blocks=2, stride=2)
        self.layer3 = make_layer(BasicBlock, self.channel[2], self.channel[3], blocks=2, stride=2)
        self.layer4 = make_layer(BasicBlock, self.channel[3], self.channel[4], blocks=2, stride=2)
        self.avgpool = base.avgpool
        self.fc = nn.Linear(base.fc.in_features, num_classes)

        self.inject1 = nn.Conv2d(3, self.channel[0], 1)  # 将pyr[1]编码为 gating
        self.inject2 = nn.Conv2d(3, self.channel[1], 1)
        self.inject3 = nn.Conv2d(3, self.channel[2], 1)
        self.inject4 = nn.Conv2d(3, self.channel[3], 1)

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局池化，保留通道维度
            nn.Sigmoid()  # 输出在 (0,1)，用于门控
        )

    def forward(self, x):
        _, pyr = lpyr.decompose(x, levels=4)
        # x = self.conv1(torch.cat([x, pyr[0]], dim=1))
        # x = self.layer1(x + self.inject1(F.interpolate(pyr[1], size=x.shape[-2:])))
        # x = self.layer2(x + self.inject2(F.interpolate(pyr[2], size=x.shape[-2:])))
        # x = self.layer3(x + self.inject3(F.interpolate(pyr[3], size=x.shape[-2:])))
        # x = self.layer4(x + self.inject4(F.interpolate(pyr[4], size=x.shape[-2:])))
        # x = self.avgpool(x)

        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        feat1 = self.inject1(F.interpolate(pyr[0], size=x.shape[-2:]))
        alpha1 = self.gate(feat1)
        x = self.layer1(x * alpha1) #这样操作似乎没有任何的精度损失(-0.15%)
        # x = self.maxpool(self.relu(self.bn1(self.conv1(pyr[0])))) #直接使用pyr[0]会导致-0.9%左右的精度损失
        # x = self.layer1(x + self.inject1(F.interpolate(pyr[1], size=x.shape[-2:]))) #直接使用pyr[1]会导致-1.5%左右的精度损失
        feat2 = self.inject2(F.interpolate(pyr[1], size=x.shape[-2:]))
        alpha2 = self.gate(feat2)
        x = self.layer2(x * alpha2)

        feat3 = self.inject3(F.interpolate(pyr[2], size=x.shape[-2:]))
        alpha3 = self.gate(feat3)
        x = self.layer3(x * alpha3)

        feat4 = self.inject4(F.interpolate(pyr[3], size=x.shape[-2:]))
        alpha4 = self.gate(feat4)
        x = self.layer4(x * alpha4)
        x = self.avgpool(x)

        return self.fc(torch.flatten(x, 1))


def train(epoch, model, criterion, optimizer):
    torch.cuda.empty_cache()
    model.train()
    # lpyr = laplacian_pyramid_simple(resolution[0], resolution[1], display_ppd, device)
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"[Epoch {epoch}] Training Loss: {running_loss / len(trainloader):.3f}")

def test(epoch, model):
    torch.cuda.empty_cache()
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    print(f"[Epoch {epoch}] Test Accuracy: {acc:.2f}%")
    return acc

# 自动调整 channel 各维度值以满足最低准确率约束
def auto_tune_channels():
    original_channel = [16, 64, 128, 256, 512]
    best_channel = original_channel.copy()
    acc_history = []

    for idx in range(len(best_channel)):
        current = best_channel[idx] * 2
        print(f"\n🔍 正在优化第 {idx} 维 channel（当前值：{current//2}）")
        while current > 3:
            # 候选新值
            candidate = current // 2
            if candidate < 3:
                break

            # 构建新模型
            candidate_channel = best_channel.copy()
            candidate_channel[idx] = candidate
            print(f" ➜  尝试 {candidate_channel}")

            model = PyramidResNet18(num_classes=10, channel_net=candidate_channel)
            model = model.to(device)
            summary(model, input_size=(3, 32, 32))

            if os.path.isfile(checkpoint_path) and load_pretrained_weights:
                print(f"⚡️ Loading pretrained weights from {checkpoint_path}")
                model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            else:
                print("No pretrained weights found, training from scratch.")

            # summary(model, input_size=(3, 32, 32))

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

            # 训练模型
            best_acc = 0
            for epoch in tqdm(range(1, 101)):
                print(f"Epoch {epoch}/100")
                train(epoch, model, criterion, optimizer)
                acc = test(epoch, model)
                if acc > best_acc:
                    best_acc = acc
                    print(f"✅ New best accuracy {best_acc:.2f}%")
                scheduler.step()

            acc_history.append((candidate_channel.copy(), best_acc))
            print(f" ✅ 尝试 channel {candidate_channel}，准确率：{best_acc:.2f}%")

            # 保存准确率到文件
            save_path = f"channel_{'_'.join(map(str, candidate_channel))}_acc_{best_acc:.2f}.txt"
            with open(save_path, 'w') as f:
                f.write(f"channel: {candidate_channel}\naccuracy: {best_acc:.2f}\n")

            if best_acc >= 94.0:
                best_channel[idx] = candidate
                current = candidate  # 尝试更小值
            else:
                print(f" ❌ {candidate} 太小，保持 {current}")
                break


    print("\n🎉 最优通道配置：", best_channel)
    return best_channel, acc_history


if __name__ == '__main__':
    best_config, history = auto_tune_channels()

# 将原本训练的RGB空间变为线性XYZ空间