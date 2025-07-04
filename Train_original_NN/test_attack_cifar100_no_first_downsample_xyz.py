import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.models import resnet18
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
from tqdm import tqdm
import numpy as np

peak_luminance = 500.0

def srgb_to_linear_rgb(srgb):
    threshold = 0.04045
    linear = torch.where(
        srgb <= threshold,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4
    )
    return linear

def linear_rgb_to_xyz(rgb):
    M = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                      [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]], dtype=rgb.dtype, device=rgb.device)
    rgb = rgb.permute(1, 2, 0)  # (C,H,W) -> (H,W,C)
    xyz = torch.tensordot(rgb, M.T, dims=1)  # (H,W,3)
    return xyz.permute(2, 0, 1)  # (3,H,W)

class RGBtoXYZTransform:
    def __init__(self, peak_luminance=500.0):
        self.peak_luminance = peak_luminance

    def __call__(self, tensor):
        tensor = srgb_to_linear_rgb(tensor)
        tensor = linear_rgb_to_xyz(tensor)
        return tensor * self.peak_luminance

# 设备
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 测试集预处理
transform_test = transforms.Compose([
    transforms.ToTensor(),
    RGBtoXYZTransform(peak_luminance=peak_luminance)
])

data_root = r'../Datasets/CIFAR10/data'
testset = torchvision.datasets.CIFAR100(root=data_root, train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

# 模型定义（和训练时一致）
model = resnet18(weights=None)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
model.fc = nn.Linear(model.fc.in_features, 100)
model = model.to(device)

# 加载训练好的权重（注意路径和文件名）
model.load_state_dict(torch.load(f'../HVS_for_better_NN_pth/best_resnet18_cifar100_no_first_downsample_xyz_pl{peak_luminance}.pth', map_location=device))
model.eval()

criterion = nn.CrossEntropyLoss()

# 使用 ART 封装模型
classifier = PyTorchClassifier(
    model=model,
    loss=criterion,
    input_shape=(3, 32, 32),
    nb_classes=100,
    optimizer=None,
    device_type='cuda' if torch.cuda.is_available() else 'cpu'
)

# PGD 攻击配置
attack = ProjectedGradientDescent(
    estimator=classifier,
    norm=np.inf,
    eps=8/255 * peak_luminance,    # 这里乘以peak_luminance保证尺度匹配（你可以根据实际情况调整）
    eps_step=2/255 * peak_luminance,
    max_iter=40,
    targeted=False,
    batch_size=100
)

# 测试干净样本准确率
def test_clean():
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(testloader, desc="Clean Testing"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    acc = 100. * correct / total
    print(f"✅ Clean Accuracy ({total} samples): {acc:.2f}%")
    return acc

# 测试PGD对抗样本准确率
def test_adversarial():
    correct = 0
    total = 0
    for inputs, targets in tqdm(testloader, desc="PGD Adversarial Testing"):
        inputs_np = inputs.cpu().numpy()
        targets_np = targets.cpu().numpy()
        # 生成对抗样本
        adv_inputs_np = attack.generate(x=inputs_np, y=targets_np)
        adv_inputs = torch.tensor(adv_inputs_np).to(device)
        targets = targets.to(device)
        outputs = model(adv_inputs)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
    acc = 100. * correct / total
    print(f"⚠️ PGD Adversarial Accuracy ({total} samples): {acc:.2f}%")
    return acc

if __name__ == "__main__":
    test_clean()
    test_adversarial()
