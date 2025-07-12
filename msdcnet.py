import torch
import torch.nn as nn
import torch.nn.functional as F

# Mish activation function
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

# Dynamic Convolution Layer with Multi-Scale Kernels
class MultiScaleDynamicConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        assert in_channels == out_channels, "Depthwise conv requires in_channels == out_channels"

        self.k3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=in_channels)
        self.k5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, groups=in_channels)
        self.k7 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3, groups=in_channels)

        self.pointwise = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, kernel_size=1),
            Mish(),
            nn.Conv2d(out_channels // 4, 3, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        b, c, h, w = x.size()
        x3 = self.k3(x)
        x5 = self.k5(x)
        x7 = self.k7(x)

        x_cat = torch.cat([x3, x5, x7], dim=1)
        x_out = self.pointwise(x_cat)

        weights = self.attn(x_out).view(b, 3, 1, 1, 1)
        x_stack = torch.stack([x3, x5, x7], dim=1)
        return (weights * x_stack).sum(dim=1)

# Squeeze-and-Excitation Block
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            Mish(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# MSDC Block with residual connection
class MSDCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.downsample = downsample or (in_channels != out_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.dynamic_conv = MultiScaleDynamicConv(out_channels, out_channels)
        self.se = SEBlock(out_channels)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = Mish()

        if self.downsample:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        x = self.conv1(x)
        x = self.dynamic_conv(x)
        x = self.se(x)
        x = self.norm(x)
        return self.act(x + identity)

# Full MSDCNet
class MSDCNet(nn.Module):
    def __init__(self, num_classes=10, drop_rate=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            Mish()
        )
        self.stage1 = MSDCBlock(32, 64, downsample=True)
        self.stage2 = MSDCBlock(64, 128, downsample=True)
        self.stage3 = MSDCBlock(128, 256, downsample=True)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(256, num_classes)
        )

        self._init_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

# Utility to freeze/unfreeze layers
def set_trainable_layers(model, trainable_names):
    for name, param in model.named_parameters():
        param.requires_grad = any(t in name for t in trainable_names)
