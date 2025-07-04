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

# ===================== 0. 定义 XYZ 空间转换函数 =====================
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
        tensor = srgb_to_linear_rgb(tensor)
        tensor = linear_rgb_to_xyz(tensor)
        return tensor * self.peak_luminance

# ===================== 1. 设置设备和路径 =====================
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_root = '../Datasets/CIFAR10/data'

# ===================== 2. 准备 sRGB & XYZ 数据变换器 =====================
transform_srgb = transforms.ToTensor()  # 用于对抗样本生成
transform_xyz = transforms.Compose([
    transforms.ToTensor(),
    RGBtoXYZTransform(peak_luminance=peak_luminance)
])  # 用于模型输入

# ===================== 3. 加载测试集（两个版本） =====================
testset_srgb = torchvision.datasets.CIFAR100(root=data_root, train=False, download=False, transform=transform_srgb)
testloader_srgb = DataLoader(testset_srgb, batch_size=100, shuffle=False, num_workers=4)

# ===================== 4. 加载训练好的模型 =====================
model = resnet18(weights=None)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
model.fc = nn.Linear(model.fc.in_features, 100)
model = model.to(device)
model.load_state_dict(torch.load(
    f'../HVS_for_better_NN_pth/best_resnet18_cifar100_no_first_downsample_xyz_pl{peak_luminance}.pth'
))
model.eval()

# ===================== 5. ART classifier 包装（使用 sRGB 空间） =====================
classifier = PyTorchClassifier(
    model=model,
    clip_values=(0.0, 1.0),
    loss=nn.CrossEntropyLoss(),
    optimizer=None,
    input_shape=(3, 32, 32),
    nb_classes=100,
    device_type='gpu' if torch.cuda.is_available() else 'cpu'
)

# ===================== 6. 准备用于攻击的数据 =====================
def get_test_data_for_attack(dataloader, n_batches=1):
    x_list, y_list = [], []
    for idx, (inputs, targets) in enumerate(dataloader):
        x_list.append(inputs)
        y_list.append(targets)
        if idx + 1 == n_batches:
            break
    x = torch.cat(x_list, dim=0)  # shape: (N,3,32,32)
    y = torch.cat(y_list, dim=0)
    return x.numpy(), y.numpy(), x

x_test_np, y_test, x_test_tensor = get_test_data_for_attack(testloader_srgb, n_batches=10)

# ===================== 7. 执行 PGD 攻击（sRGB 空间） =====================
attack = ProjectedGradientDescent(
    estimator=classifier,
    eps=0.1,
    eps_step=0.01,
    max_iter=32,
    verbose=True
)
x_adv_np = attack.generate(x=x_test_np)
x_adv_tensor = torch.tensor(x_adv_np)

# ===================== 8. 转换原始 & 对抗样本为 XYZ 空间 =====================
rgb_to_xyz = RGBtoXYZTransform(peak_luminance=peak_luminance)
x_test_xyz = torch.stack([rgb_to_xyz(x) for x in x_test_tensor])
x_adv_xyz = torch.stack([rgb_to_xyz(x) for x in x_adv_tensor])

# ===================== 9. 评估模型准确率 =====================
model.eval()
with torch.no_grad():
    logits_clean = model(x_test_xyz.to(device))
    logits_adv = model(x_adv_xyz.to(device))

pred_clean = logits_clean.argmax(dim=1).cpu().numpy()
pred_adv = logits_adv.argmax(dim=1).cpu().numpy()

acc_clean = np.mean(pred_clean == y_test)
acc_adv = np.mean(pred_adv == y_test)

print(f"\n✅ Clean Accuracy (1000 samples): {acc_clean * 100:.2f}%")
print(f"⚠️ PGD Adversarial Accuracy (1000 samples): {acc_adv * 100:.2f}%")
