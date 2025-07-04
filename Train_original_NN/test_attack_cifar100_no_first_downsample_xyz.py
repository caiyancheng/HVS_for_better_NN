import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader

import numpy as np
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
from tqdm import tqdm

peak_luminance = 500.0

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

# ===================== 1. 设置设备 & 数据集路径 =====================
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_root = '../Datasets/CIFAR10/data'

# ===================== 2. 数据预处理 =====================
transform_test = transforms.Compose([
    transforms.ToTensor(),
    RGBtoXYZTransform(peak_luminance=peak_luminance)
])

testset = torchvision.datasets.CIFAR100(root=data_root, train=False, download=False, transform=transform_test)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

# ===================== 3. 加载训练好的模型结构和参数 =====================
model = resnet18(weights=None)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
model.fc = nn.Linear(model.fc.in_features, 100)
model = model.to(device)
model.load_state_dict(torch.load(f'../HVS_for_better_NN_pth/best_resnet18_cifar100_no_first_downsample_xyz_pl{peak_luminance}.pth'))
model.eval()

# ===================== 4. 包装 ART 的 PyTorchClassifier =====================
criterion = nn.CrossEntropyLoss()

classifier = PyTorchClassifier(
    model=model,
    clip_values=(0.0, 1.0),  # 由于 transform 只是 ToTensor，没有标准化
    loss=criterion,
    optimizer=None,  # 攻击不需要训练优化器
    input_shape=(3, 32, 32),
    nb_classes=100,
    device_type='gpu' if torch.cuda.is_available() else 'cpu'
)

# ===================== 5. 准备测试数据一批用于攻击 =====================
def get_numpy_test_data(dataloader, n_batches=1):
    x_list, y_list = [], []
    for idx, (inputs, targets) in enumerate(dataloader):
        x_list.append(inputs)
        y_list.append(targets)
        if idx + 1 == n_batches:
            break
    x = torch.cat(x_list, dim=0).numpy()
    y = torch.cat(y_list, dim=0).numpy()
    return x, y

x_test, y_test = get_numpy_test_data(testloader, n_batches=10)  # 取前1000个样本

# ===================== 6. 执行 PGD 对抗攻击 =====================
attack = ProjectedGradientDescent(
    estimator=classifier,
    eps=8/255,         # 最大扰动
    eps_step=2/255,    # 每步的扰动大小
    max_iter=40,
    verbose=True
)

x_adv = attack.generate(x=x_test)

# ===================== 7. 在干净样本和对抗样本上评估准确率 =====================
pred_clean = np.argmax(classifier.predict(x_test), axis=1)
pred_adv = np.argmax(classifier.predict(x_adv), axis=1)

acc_clean = np.mean(pred_clean == y_test)
acc_adv = np.mean(pred_adv == y_test)

print(f"\n✅ Clean Accuracy (1000 samples): {acc_clean * 100:.2f}%") #76.8%
print(f"⚠️ PGD Adversarial Accuracy (1000 samples): {acc_adv * 100:.2f}%") #6.5%
