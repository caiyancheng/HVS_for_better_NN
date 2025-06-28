import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class GaussianBlurLayer(nn.Module):
    def __init__(self, channels, kernel_size=5, sigma=1.0):
        super().__init__()
        # 构造高斯核，固定参数
        self.channels = channels
        self.kernel_size = kernel_size
        self.sigma = sigma

        # 创建高斯核
        kernel = self._create_gaussian_kernel(kernel_size, sigma)
        kernel = kernel.expand(channels, 1, kernel_size, kernel_size)
        self.register_buffer('weight', kernel)
        self.groups = channels

    def _create_gaussian_kernel(self, kernel_size, sigma):
        import math
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        # 分组卷积实现通道独立高斯模糊
        return F.conv2d(x, weight=self.weight, padding=self.kernel_size // 2, groups=self.groups)

class LaplacianPyramidBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.gauss1 = GaussianBlurLayer(in_channels, kernel_size=5, sigma=1.0)
        self.gauss2 = GaussianBlurLayer(in_channels, kernel_size=5, sigma=2.0)
        self.gauss3 = GaussianBlurLayer(in_channels, kernel_size=5, sigma=4.0)

        # 对应的下采样卷积层（代替传统的卷积下采样）
        self.conv1 = nn.Conv2d(in_channels, out_channels//4, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels//4, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels//2, kernel_size=3, stride=4, padding=1)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

        # 用残差连接调整维度
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.residual_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # 原始图像的不同高斯模糊尺度
        g1 = self.gauss1(x)  # scale 1
        g2 = self.gauss2(x)  # scale 2
        g3 = self.gauss3(x)  # scale 3

        # 下采样
        c1 = self.conv1(g1)  # 32x32
        c2 = self.conv2(g2)  # 16x16
        c3 = self.conv3(g3)  # 8x8

        # 上采样回最高分辨率，方便融合
        c2_up = F.interpolate(c2, size=c1.shape[2:], mode='bilinear', align_corners=False)
        c3_up = F.interpolate(c3, size=c1.shape[2:], mode='bilinear', align_corners=False)

        # 融合不同尺度特征
        out = torch.cat([c1, c2_up, c3_up], dim=1)  # 通道拼接
        out = self.bn(out)
        out = self.relu(out)

        # 残差连接（通道对齐）
        res = self.residual_conv(x)
        res = self.residual_bn(res)

        out = out + res
        out = self.relu(out)

        return out

class CustomResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.lap_pyr_block = LaplacianPyramidBlock(in_channels=3, out_channels=64)
        # 加载预训练ResNet18骨架
        backbone = resnet18(weights=None)

        # 保留ResNet18的layer2,3,4 和fc
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.avgpool = backbone.avgpool
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.lap_pyr_block(x)     # 替代前半部分
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# 替换训练脚本中的模型初始化为：
# model = CustomResNet18(num_classes=10).to(device)
