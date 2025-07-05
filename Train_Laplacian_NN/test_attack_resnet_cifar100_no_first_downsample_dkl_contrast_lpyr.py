import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from lpyr_dec import *
import numpy as np
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
from tqdm import tqdm
from torchvision.models.resnet import BasicBlock
import torch.nn.functional as F
import argparse

parser = argparse.ArgumentParser(description='Run training with customizable peak luminance and display size.')
parser.add_argument('--pyr_levels', type=int, default=4)
parser.add_argument('--eps_value', type=float, default=0.1)
parser.add_argument('--peak_luminance', type=float, default=500.0, help='Peak luminance value (default: 500.0)')
parser.add_argument('--diagonal_size_inches', type=float, default=10.0, help='Display diagonal size in inches (default: 10.0)')
args = parser.parse_args()

pyr_levels = args.pyr_levels
peak_luminance = args.peak_luminance
resolution = [32, 32]
diagonal_size_inches = args.diagonal_size_inches
viewing_distance_meters = 1
ar = resolution[0]/resolution[1]
height_mm = math.sqrt( (diagonal_size_inches*25.4)**2 / (1+ar**2) )
display_size_m = (ar*height_mm/1000, height_mm/1000)
pix_deg = 2 * math.degrees(math.atan(0.5 * display_size_m[0] / resolution[0] / viewing_distance_meters))
display_ppd = 1 / pix_deg
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lpyr = laplacian_pyramid_simple_contrast(resolution[1], resolution[0], display_ppd, device, contrast='weber_g1')

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

# ===================== 1. 设置设备和路径 =====================
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_root = '../Datasets/CIFAR10/data'

# ===================== 2. 准备 sRGB & DKL 数据变换器 =====================
transform_srgb = transforms.ToTensor()  # 用于对抗样本生成
transform_dkl = transforms.Compose([
    transforms.ToTensor(),
    RGBtoDKLTransform(peak_luminance=peak_luminance)
])  # 用于模型输入

# ===================== 3. 加载测试集（两个版本） =====================
testset_srgb = torchvision.datasets.CIFAR100(root=data_root, train=False, download=False, transform=transform_srgb)
testloader_srgb = DataLoader(testset_srgb, batch_size=100, shuffle=False, num_workers=4)

# ===================== 4. 加载训练好的模型 =====================
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
    def __init__(self, num_classes=10):
        super().__init__()
        base = resnet18(weights=None)
        # self.channel = [16, 64, 128, 256, 256] #最初是[64, 64, 128, 256, 512] #64-93.90%, 32-94.03%, 16-94.39%， 8-94.16%, 4-94.04%
        self.channel = [64, 64, 128, 256, 512]
        # 最后一块变成256好像对精度也没啥影响93.95%的准确度左右
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
        self.fc = nn.Linear(self.channel[4], num_classes)

        self.inject1 = nn.Conv2d(3, self.channel[0], 1)  # 将pyr[1]编码为 gating
        self.inject2 = nn.Conv2d(3, self.channel[1], 1)
        self.inject3 = nn.Conv2d(3, self.channel[2], 1)
        self.inject4 = nn.Conv2d(3, self.channel[3], 1)

        self.gate = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),  # 全局池化，保留通道维度
            nn.Sigmoid()  # 输出在 (0,1)，用于门控
        )

    def forward(self, x):
        pyr, _ = lpyr.decompose(x, levels=pyr_levels)
        # x = self.conv1(torch.cat([x, pyr[0]], dim=1))
        # x = self.layer1(x + self.inject1(F.interpolate(pyr[1], size=x.shape[-2:])))
        # x = self.layer2(x + self.inject2(F.interpolate(pyr[2], size=x.shape[-2:])))
        # x = self.layer3(x + self.inject3(F.interpolate(pyr[3], size=x.shape[-2:])))
        # x = self.layer4(x + self.inject4(F.interpolate(pyr[4], size=x.shape[-2:])))
        # x = self.avgpool(x)

        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        feat1 = self.inject1(F.interpolate(pyr[0], size=x.shape[-2:]))  #[B, 64, 32, 32]
        alpha1 = self.gate(feat1) #[B, 64, 1, 1]
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


model = PyramidResNet18(num_classes=100)
model = model.to(device)
model.load_state_dict(torch.load(
f'../HVS_for_better_NN_pth/best_resnet18_cifar100_dkl_contrast_lpyr_level_{pyr_levels}_pl{peak_luminance}_dsi{diagonal_size_inches}_1.pth'
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

x_test_np, y_test, x_test_tensor = get_test_data_for_attack(testloader_srgb, n_batches=100)

# ===================== 7. 执行 PGD 攻击（sRGB 空间） =====================
eps_value = args.eps_value#0.1 #0.02
attack = ProjectedGradientDescent(
    estimator=classifier,
    eps=eps_value,
    eps_step=eps_value*0.1,
    max_iter=32,
    verbose=True
)
x_adv_np = attack.generate(x=x_test_np)
x_adv_tensor = torch.tensor(x_adv_np)

# ===================== 8. 转换原始 & 对抗样本为 DKL 空间 =====================
rgb_to_dkl = RGBtoDKLTransform(peak_luminance=peak_luminance)
x_test_dkl = torch.stack([rgb_to_dkl(x) for x in x_test_tensor])
x_adv_dkl = torch.stack([rgb_to_dkl(x) for x in x_adv_tensor])

# ===================== 9. 评估模型准确率 =====================
model.eval()
with torch.no_grad():
    logits_clean = model(x_test_dkl.to(device))
    logits_adv = model(x_adv_dkl.to(device))

pred_clean = logits_clean.argmax(dim=1).cpu().numpy()
pred_adv = logits_adv.argmax(dim=1).cpu().numpy()

acc_clean = np.mean(pred_clean == y_test)
acc_adv = np.mean(pred_adv == y_test)

print(f"\n✅ Clean Accuracy (10000 samples): {acc_clean * 100:.2f}%") #[100: 75.08%; 500: 74.44%];
print(f"⚠️ PGD Adversarial Accuracy (10000 samples): {acc_adv * 100:.2f}%") #0.1: [100: 16.22%; 500: 17.42%]; 0.02: [100: 39.47%; 500: 43.88%] DKL space的准确率好像高得多？
