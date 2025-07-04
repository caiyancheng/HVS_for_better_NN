import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm
from torchsummary import summary
from lpyr_dec import *
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock
from csf import castleCSF
import json
import random
import os
from torchvision.transforms import GaussianBlur
from torch.functional import Tensor



os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='Run training with customizable peak luminance and display size.')
parser.add_argument('--pyr_levels', type=int, default=4)
parser.add_argument('--peak_luminance', type=float, default=500.0, help='Peak luminance value (default: 500.0)')
parser.add_argument('--diagonal_size_inches', type=float, default=10.0, help='Display diagonal size in inches (default: 10.0)')
args = parser.parse_args()
# peak_luminance = args.peak_luminance
# diagonal_size_inches = args.diagonal_size_inches

def set_seed(seed=42):
    random.seed(seed)  # Python 原生随机模块
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # 当前 GPU
    torch.cuda.manual_seed_all(seed)  # 所有 GPU（多卡）

    torch.backends.cudnn.deterministic = True  # 保证每次卷积结果一样（可能稍慢）
    torch.backends.cudnn.benchmark = False  # 关闭自动优化卷积算法选择（可复现）


set_seed(66)  # 可改成你喜欢的种子数

pyr_levels = args.pyr_levels #4
# Viewing Condition Setting
peak_luminance = args.peak_luminance #500.0
diagonal_size_inches = args.diagonal_size_inches #10  # 5
checkpoint_path = f'../HVS_for_better_NN_pth/best_resnet18_cifar100_dkl_CSF_2_level{pyr_levels}_pl{peak_luminance}_dsi{diagonal_size_inches}_1.pth'
print(checkpoint_path)
load_pretrained_weights = False
resolution = [32, 32]
viewing_distance_meters = 1

import sys

class Logger(object):
    def __init__(self, filename="training_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(f"../training_log/training_resnet18_cifar100_dkl_CSF_2_level{pyr_levels}_pl{peak_luminance}_dsi{diagonal_size_inches}.txt")
sys.stderr = sys.stdout  # 错误输出也重定向到同一个文件

ar = resolution[0] / resolution[1]
height_mm = math.sqrt((diagonal_size_inches * 25.4) ** 2 / (1 + ar ** 2))
display_size_m = (ar * height_mm / 1000, height_mm / 1000)
pix_deg = 2 * math.degrees(math.atan(0.5 * display_size_m[0] / resolution[0] / viewing_distance_meters))
display_ppd = 1 / pix_deg

device = 'cuda' if torch.cuda.is_available() else 'cpu'
lpyr = laplacian_pyramid_simple_contrast(resolution[1], resolution[0], display_ppd, device, contrast='weber_g1')

with open('Train_Laplacian_NN/cvvdp_parameters.json', 'r') as fp:
    parameters = json.load(fp)
CSF_castleCSF = castleCSF(csf_version=parameters['csf'], device=device)
csf_sigma = torch.as_tensor(parameters['csf_sigma'], device=device)
sensitivity_correction = torch.as_tensor(parameters['sensitivity_correction'], device=device)
mask_p = torch.as_tensor(parameters['mask_p'], device=device)
mask_q = torch.as_tensor(parameters['mask_q'], device=device)
mask_c = torch.as_tensor(parameters['mask_c'], device=device)
pu_dilate = parameters['pu_dilate']
pu_blur = GaussianBlur(int(pu_dilate * 4) + 1, pu_dilate)
pu_padsize = int(pu_dilate * 2)
do_xchannel_masking = True if parameters['xchannel_masking'] == "on" else False
d_max = torch.as_tensor(parameters['d_max'], device=device)

# --- ✅ DKL转换相关矩阵 ---
LMS2006_to_DKLd65 = torch.tensor([
    [1.000000000000000, 1.000000000000000, 0],
    [1.000000000000000, -2.311130179947035, 0],
    [-1.000000000000000, -1.000000000000000, 50.977571328718781]
], dtype=torch.float32)
XYZ_to_LMS2006 = torch.tensor([
    [0.187596268556126, 0.585168649077728, -0.026384263306304],
    [-0.133397430663221, 0.405505777260049, 0.034502127690364],
    [0.000244379021663, -0.000542995890619, 0.019406849066323]
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
        tensor = srgb_to_linear_rgb(tensor)  # sRGB → linear RGB
        tensor = linear_rgb_to_xyz(tensor) * self.peak_luminance  # → XYZ (cd/m²)
        tensor = xyz_to_lms2006(tensor)  # → LMS
        tensor = lms_to_dkl(tensor)  # → DKL
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

trainset = torchvision.datasets.CIFAR100(root=data_root, train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR100(root=data_root, train=False, download=False, transform=transform_test)
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


def phase_uncertainty(M):
    # Blur only when the image is larger then the required pad size
    if pu_dilate != 0 and M.shape[-2] > pu_padsize and M.shape[-1] > pu_padsize:
        # M_pu = utils.imgaussfilt( M, self.pu_dilate ) * torch.pow(10.0, self.mask_c)
        M_pu = pu_blur.forward(M) * (10 ** mask_c)
    else:
        M_pu = M * (10 ** mask_c)
    return M_pu


def mask_pool(C):
    # Cross-channel masking
    num_ch = C.shape[0]
    xcm_weights = torch.as_tensor(parameters['xcm_weights'], device=device, dtype=torch.float32)
    if do_xchannel_masking:
        M = torch.empty_like(C)
        xcm_weights = torch.reshape((2 ** xcm_weights), (4, 4, 1, 1, 1))[:num_ch, ...]
        for cc in range(num_ch):  # for each channel: Sust, RG, VY, Trans
            M[cc, ...] = torch.sum(C * xcm_weights[:, cc], dim=0, keepdim=True)
    else:
        cm_weights = torch.reshape((2 ** xcm_weights), (4, 1, 1, 1))[:num_ch, ...]
        M = C * cm_weights
    return M


def safe_pow(x: Tensor, p):
    # assert (not x.isnan().any()) and (not x.isinf().any()), "Must not be nan"
    # assert torch.all(x>=0), "Must be positive"

    if True:  # isinstance( p, Tensor ) and p.requires_grad:
        # If we need a derivative with respect to p, x must not be 0
        epsilon = torch.as_tensor(0.00001, device=x.device)
        return (x + epsilon) ** p - epsilon ** p
    else:
        return x ** p

def pow_neg( x:Tensor, p ):
    #assert (not x.isnan().any()) and (not x.isinf().any()), "Must not be nan"

    #return torch.tanh(100*x) * (torch.abs(x) ** p)

    min_v = torch.as_tensor( 0.00001, device=x.device )
    return (torch.max(x,min_v) ** p) + (torch.max(-x,min_v) ** p) - min_v**p

def cm_transd(C_p):
    num_ch = C_p.shape[0]
    p = mask_p
    q = mask_q[0:num_ch].view(num_ch, 1, 1, 1)
    M = phase_uncertainty(mask_pool(safe_pow(torch.abs(C_p), q)))
    D_max = 10 ** d_max
    return D_max * pow_neg(C_p, p) / (0.2 + M)


def apply_masking_model(T, S):
    num_ch = T.shape[0]
    ch_gain = torch.reshape(torch.as_tensor([1, 1.45, 1, 1.], device=T.device), (4, 1, 1, 1))[:num_ch, ...]
    T_p = T * S * ch_gain
    D = torch.abs(cm_transd(T_p))
    return D


class PyramidResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        base = resnet18(weights=None)
        # self.channel = [16, 64, 128, 256, 256] #最初是[64, 64, 128, 256, 512] #64-93.90%, 32-94.03%, 16-94.39%， 8-94.16%, 4-94.04%
        self.channel = [64, 64, 128, 256, 512]
        # 最后一块变成256好像对精度也没啥影响93.95%的准确度左右
        # base.conv1 = nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 输入 concat image + L0
        base.conv1 = nn.Conv2d(3, self.channel[0], kernel_size=3, stride=1, padding=1,
                               bias=False)  # 输入 concat image + L0
        base.maxpool = nn.Identity()

        self.conv1 = base.conv1
        self.bn1 = nn.BatchNorm2d(self.channel[0])  ###卧槽！反而+0.4%的正向增长
        self.relu = base.relu
        self.maxpool = base.maxpool
        # self.layer1 = base.layer1

        self.layer1 = make_layer(BasicBlock, self.channel[0], self.channel[1], blocks=2, stride=1)
        self.layer2 = make_layer(BasicBlock, self.channel[1], self.channel[2], blocks=2, stride=2)
        self.layer3 = make_layer(BasicBlock, self.channel[2], self.channel[3], blocks=2, stride=2)
        self.layer4 = make_layer(BasicBlock, self.channel[3], self.channel[4], blocks=2, stride=2)
        self.avgpool = base.avgpool
        self.fc = nn.Linear(self.channel[4], num_classes)

        self.inject1 = nn.Conv2d(3, 1, 1)  # 将pyr[1]编码为 gating
        self.inject2 = nn.Conv2d(3, 1, 1)
        self.inject3 = nn.Conv2d(3, 1, 1)
        self.inject4 = nn.Conv2d(3, 1, 1)

        # self.gate = nn.Sequential(
        #     # nn.AdaptiveAvgPool2d(1),  # 全局池化，保留通道维度
        #     nn.Sigmoid()  # 输出在 (0,1)，用于门控
        # )

    def forward(self, x):
        lpyr_contrast, L_bkg_pyr = lpyr.decompose(x, levels=pyr_levels)
        rho_band = lpyr.get_freqs()
        PYR_list = []
        for bb in range(len(lpyr_contrast)):
            is_baseband = (bb == (len(lpyr_contrast) - 1))
            if bb == (len(lpyr_contrast) - 1) or bb == 0:
                pyr_level = lpyr_contrast[bb] * 2 #[128,3,32,32]
                logL_bkg = L_bkg_pyr[bb] #* 2
            else:
                pyr_level = lpyr_contrast[bb]
                logL_bkg = L_bkg_pyr[bb]
            rho = rho_band[bb]
            ch_height, ch_width = logL_bkg.shape[-2], logL_bkg.shape[-1]
            S = torch.empty((3, x.shape[0],ch_height, ch_width), device=device)
            for cc in range(3):
                cch = cc if cc < 3 else 0  # Y, rg, yv
                # The sensitivity is always extracted for the reference frame
                S[cc, :, :, :] = CSF_castleCSF.sensitivity(rho, 0, logL_bkg[..., 0, :, :], cch, csf_sigma) * 10.0 ** (
                        sensitivity_correction / 20.0)  # 第一个的平均值应该为2.5645
            if is_baseband:
                D = (torch.abs(pyr_level.permute(1, 0, 2, 3)) * S)
            else:
                # dimensions: [channel,frame,height,width]
                D = (torch.abs(pyr_level.permute(1, 0, 2, 3)) * S)
                # D = apply_masking_model(pyr_level.permute(1, 0, 2, 3), S)
            PYR_list.append(D.permute(1, 0, 2, 3))
        # x = self.conv1(torch.cat([x, pyr[0]], dim=1))
        # x = self.layer1(x + self.inject1(F.interpolate(pyr[1], size=x.shape[-2:])))
        # x = self.layer2(x + self.inject2(F.interpolate(pyr[2], size=x.shape[-2:])))
        # x = self.layer3(x + self.inject3(F.interpolate(pyr[3], size=x.shape[-2:])))
        # x = self.layer4(x + self.inject4(F.interpolate(pyr[4], size=x.shape[-2:])))
        # x = self.avgpool(x)

        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        alpha1 = self.inject1(F.interpolate(PYR_list[0], size=x.shape[-2:]))  # [B, 64, 32, 32]
        # alpha1 = self.gate(feat1)  # [B, 64, 1, 1]
        x = self.layer1(x * alpha1)  # 这样操作似乎没有任何的精度损失(-0.15%)
        # x = self.maxpool(self.relu(self.bn1(self.conv1(pyr[0])))) #直接使用pyr[0]会导致-0.9%左右的精度损失
        # x = self.layer1(x + self.inject1(F.interpolate(pyr[1], size=x.shape[-2:]))) #直接使用pyr[1]会导致-1.5%左右的精度损失
        alpha2 = self.inject2(F.interpolate(PYR_list[1], size=x.shape[-2:]))
        # alpha2 = self.gate(feat2)
        x = self.layer2(x * alpha2)

        alpha3 = self.inject3(F.interpolate(PYR_list[2], size=x.shape[-2:]))
        # alpha3 = self.gate(feat3)
        x = self.layer3(x * alpha3)

        alpha4 = self.inject4(F.interpolate(PYR_list[3], size=x.shape[-2:]))
        # alpha4 = self.gate(feat4)
        x = self.layer4(x * alpha4)
        x = self.avgpool(x)

        return self.fc(torch.flatten(x, 1))


# model = resnet18(weights=None)
# model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 3x3 conv
# model.maxpool = nn.Identity()  # 取消 maxpool
# model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR-10 有10类
model = PyramidResNet18(num_classes=100)
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


def train(epoch):
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


def test(epoch):
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


if __name__ == '__main__':
    best_acc = 0
    for epoch in tqdm(range(1, 101)):  # 可调节 epoch
        train(epoch)
        acc = test(epoch)
        scheduler.step()
        # 保存最好模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✅ Saved best model with accuracy {best_acc:.2f}%")

# 准确率只有74.05%, 有点点问题; 此处的transducer在masking的输入处只有单一test,根本无reference