import torch
from torch import nn
import torch.nn.functional as F

class MaxFiltering(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int = 3, tau: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.norm = nn.GroupNorm(32, in_channels)
        self.nonlinear = nn.ReLU()
        self.max_pool = nn.MaxPool3d(
            kernel_size=(tau + 1, kernel_size, kernel_size),
            padding=(tau // 2, kernel_size // 2, kernel_size // 2),
            stride=1
        )
        self.margin = tau // 2  # 1

    def forward(self, inputs):
        features = []
        for l, x in enumerate(inputs):  # 首先执行self.conv
            features.append(self.conv(x))

        outputs = []
        for l, x in enumerate(features):
            func = lambda f: F.interpolate(f, size=x.shape[2:], mode="bilinear", align_corners=False)  # 定义了一个函数，执行线性插值
            feature_3d = []
            for k in range(max(0, l - self.margin), min(len(features), l + self.margin + 1)):  # 0~1, 0~2, 1~3, 2~4, 3~4
                feature_3d.append(func(features[k]) if k != l else features[k])
            feature_3d = torch.stack(feature_3d, dim=2)
            max_pool = self.max_pool(feature_3d)[:, :, min(l, self.margin)]  # 进行3dmax
            output = max_pool + inputs[l]  # 与输入相加
            outputs.append(self.nonlinear(self.norm(output)))
        return outputs
