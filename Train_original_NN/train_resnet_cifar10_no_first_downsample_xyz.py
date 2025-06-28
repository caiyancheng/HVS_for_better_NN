import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
import os
from tqdm import tqdm
from torchsummary import summary

def srgb_to_linear_rgb(srgb):
    """sRGB 转线性 RGB (gamma 解码)，输入 [0, 1]，输出 [0, 1]"""
    threshold = 0.04045
    linear = torch.where(
        srgb <= threshold,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4
    )
    return linear

def linear_rgb_to_xyz(rgb):
    """线性 RGB → XYZ，D65"""
    # 使用 sRGB 的 D65 到 XYZ 转换矩阵
    M = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                      [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]], dtype=rgb.dtype, device=rgb.device)
    rgb = rgb.permute(1, 2, 0)  # (C,H,W) → (H,W,C)
    xyz = torch.tensordot(rgb, M.T, dims=1)  # (H,W,3)
    return xyz.permute(2, 0, 1)  # (3,H,W)

class RGBtoXYZTransform:
    def __init__(self, peak_luminance=500.0):
        self.peak_luminance = peak_luminance

    def __call__(self, tensor):
        # 输入 tensor 是 [0,1] 的 sRGB，shape: (C,H,W)
        tensor = srgb_to_linear_rgb(tensor)
        tensor = linear_rgb_to_xyz(tensor)
        # 归一化 XYZ 空间，使其最大值为 1（以峰值亮度 500cd/m² 表示）
        return tensor * self.peak_luminance


# 替换为原始字符串避免 warning
# data_root = r'E:\Datasets\CIFAR10\data'
data_root = r'../Datasets/CIFAR10/data'

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    RGBtoXYZTransform(peak_luminance=500.0)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    RGBtoXYZTransform(peak_luminance=500.0)
])


trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model = resnet18(weights=None)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 3x3 conv
model.maxpool = nn.Identity()  # 取消 maxpool
model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR-10 有10类
model = model.to(device)
summary(model, input_size=(3, 32, 32))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

def train(epoch):
    model.train()
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

def test(epoch):
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

if __name__ == '__main__':
    best_acc = 0
    for epoch in tqdm(range(1, 101)):  # 可调节 epoch
        train(epoch)
        acc = test(epoch)
        scheduler.step()
        # 保存最好模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), '../HVS_for_better_NN_pth/best_resnet18_cifar10_no_first_downsample_xyz.pth')
            print(f"✅ Saved best model with accuracy {best_acc:.2f}%")
