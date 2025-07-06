import torch
from torchvision.models import resnet18
import torch.nn as nn
from torchvision.models.resnet import BasicBlock
from lpyr_dec import *
import torch.nn.functional as F
import json
from csf import castleCSF
from torchvision.transforms import GaussianBlur
from torch.functional import Tensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open('Train_Laplacian_NN/cvvdp_parameters_transducer.json', 'r') as fp:
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
d_max = torch.as_tensor(parameters['dclamp_par'], device=device)

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

class ResNet18_lpyr(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        base = resnet18(weights=None)
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 输入 concat image + L0
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

        self.inject1 = nn.Conv2d(3, 64, 1)  # 将pyr[1]编码为 gating
        self.inject2 = nn.Conv2d(3, 64, 1)
        self.inject3 = nn.Conv2d(3, 128, 1)
        self.inject4 = nn.Conv2d(3, 256, 1)

        self.gate = nn.Sigmoid()

    def set_lpyr(self, lpyr, pyr_levels):
        self.lpyr = lpyr
        self.pyr_levels = pyr_levels

    def forward(self, x):
        _, pyr = self.lpyr.decompose(x, levels=self.pyr_levels)
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        feat1 = self.inject1(F.interpolate(pyr[0], size=x.shape[-2:]))
        alpha1 = self.gate(feat1)
        x = self.layer1(x * alpha1)
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


class ResNet18_clpyr(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        base = resnet18(weights=None)
        self.channel = [64, 64, 128, 256, 512]
        base.conv1 = nn.Conv2d(3, self.channel[0], kernel_size=3, stride=1, padding=1, bias=False)  # 输入 concat image + L0
        base.maxpool = nn.Identity()

        self.conv1 = base.conv1
        self.bn1 = nn.BatchNorm2d(self.channel[0]) ###卧槽！反而+0.4%的正向增长
        self.relu = base.relu
        self.maxpool = base.maxpool

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

        self.gate = nn.Sigmoid()  # 输出在 (0,1)

    def set_lpyr(self, lpyr, pyr_levels):
        self.lpyr = lpyr
        self.pyr_levels = pyr_levels

    def forward(self, x):
        pyr, _ = self.lpyr.decompose(x, levels=self.pyr_levels)
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        feat1 = self.inject1(F.interpolate(pyr[0], size=x.shape[-2:]))  #[B, 64, 32, 32]
        alpha1 = self.gate(feat1) #[B, 64, 1, 1]
        x = self.layer1(x * alpha1) #这样操作似乎没有任何的精度损失(-0.15%)
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

class ResNet18_clpyr_CSF(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        base = resnet18(weights=None)
        self.channel = [64, 64, 128, 256, 512]
        base.conv1 = nn.Conv2d(3, self.channel[0], kernel_size=3, stride=1, padding=1,
                               bias=False)  # 输入 concat image + L0
        base.maxpool = nn.Identity()

        self.conv1 = base.conv1
        self.bn1 = nn.BatchNorm2d(self.channel[0])  ###卧槽！反而+0.4%的正向增长
        self.relu = base.relu
        self.maxpool = base.maxpool

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

        self.gate = nn.Sigmoid()

    def set_lpyr(self, lpyr, pyr_levels):
        self.lpyr = lpyr
        self.pyr_levels = pyr_levels

    def forward(self, x):
        lpyr_contrast, L_bkg_pyr = self.lpyr.decompose(x, levels=self.pyr_levels)
        rho_band = self.lpyr.get_freqs()
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
                D = (torch.abs(pyr_level.permute(1, 0, 2, 3)) * S)
            PYR_list.append(D.permute(1, 0, 2, 3))

        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        feat1 = self.inject1(F.interpolate(PYR_list[0], size=x.shape[-2:]))  # [B, 64, 32, 32]
        alpha1 = self.gate(feat1)  # [B, 64, 1, 1]
        x = self.layer1(x * alpha1)  # 这样操作似乎没有任何的精度损失(-0.15%)
        feat2 = self.inject2(F.interpolate(PYR_list[1], size=x.shape[-2:]))
        alpha2 = self.gate(feat2)
        x = self.layer2(x * alpha2)

        feat3 = self.inject3(F.interpolate(PYR_list[2], size=x.shape[-2:]))
        alpha3 = self.gate(feat3)
        x = self.layer3(x * alpha3)

        feat4 = self.inject4(F.interpolate(PYR_list[3], size=x.shape[-2:]))
        alpha4 = self.gate(feat4)
        x = self.layer4(x * alpha4)
        x = self.avgpool(x)

        return self.fc(torch.flatten(x, 1))

class ResNet18_clpyr_masking_transducer(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        base = resnet18(weights=None)
        self.channel = [64, 64, 128, 256, 512]
        base.conv1 = nn.Conv2d(3, self.channel[0], kernel_size=3, stride=1, padding=1,
                               bias=False)  # 输入 concat image + L0
        base.maxpool = nn.Identity()

        self.conv1 = base.conv1
        self.bn1 = nn.BatchNorm2d(self.channel[0])  ###卧槽！反而+0.4%的正向增长
        self.relu = base.relu
        self.maxpool = base.maxpool

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

        self.gate = nn.Sigmoid()

    def set_lpyr(self, lpyr, pyr_levels):
        self.lpyr = lpyr
        self.pyr_levels = pyr_levels

    def forward(self, x):
        lpyr_contrast, L_bkg_pyr = self.lpyr.decompose(x, levels=self.pyr_levels)
        rho_band = self.lpyr.get_freqs()
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
                D = apply_masking_model(pyr_level.permute(1, 0, 2, 3), S)
                # D = (torch.abs(pyr_level.permute(1, 0, 2, 3)) * S)
            PYR_list.append(D.permute(1, 0, 2, 3))

        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        feat1 = self.inject1(F.interpolate(PYR_list[0], size=x.shape[-2:]))  # [B, 64, 32, 32]
        alpha1 = self.gate(feat1)  # [B, 64, 1, 1]
        x = self.layer1(x * alpha1)  # 这样操作似乎没有任何的精度损失(-0.15%)
        feat2 = self.inject2(F.interpolate(PYR_list[1], size=x.shape[-2:]))
        alpha2 = self.gate(feat2)
        x = self.layer2(x * alpha2)

        feat3 = self.inject3(F.interpolate(PYR_list[2], size=x.shape[-2:]))
        alpha3 = self.gate(feat3)
        x = self.layer3(x * alpha3)

        feat4 = self.inject4(F.interpolate(PYR_list[3], size=x.shape[-2:]))
        alpha4 = self.gate(feat4)
        x = self.layer4(x * alpha4)
        x = self.avgpool(x)

        return self.fc(torch.flatten(x, 1))

def model_create(model_name, dataset_name):
    if model_name == 'resnet18':
        model = resnet18(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        if dataset_name == 'CIFAR-100':
            model.fc = nn.Linear(model.fc.in_features, 100)
        elif dataset_name == 'CIFAR-10':
            model.fc = nn.Linear(model.fc.in_features, 10)
        return model
    elif model_name == 'resnet18-lpyr':
        if dataset_name == 'CIFAR-100':
            model = ResNet18_lpyr(num_classes=100)
        elif dataset_name == 'CIFAR-10':
            model = ResNet18_lpyr(num_classes=10)
        return model
    elif model_name == 'resnet18-clpyr':
        if dataset_name == 'CIFAR-100':
            model = ResNet18_clpyr(num_classes=100)
        elif dataset_name == 'CIFAR-10':
            model = ResNet18_clpyr(num_classes=10)
        return model
    elif model_name == 'resnet18-clpyr-CSF':
        if dataset_name == 'CIFAR-100':
            model = ResNet18_clpyr_CSF(num_classes=100)
        elif dataset_name == 'CIFAR-10':
            model = ResNet18_clpyr_CSF(num_classes=10)
        return model
    elif model_name == 'resnet18-clpyr-CM-transducer':
        if dataset_name == 'CIFAR-100':
            model = ResNet18_clpyr_masking_transducer(num_classes=100)
        elif dataset_name == 'CIFAR-10':
            model = ResNet18_clpyr_masking_transducer(num_classes=10)
        return model
    else:
        raise NotImplementedError('The setting is not implemented.')