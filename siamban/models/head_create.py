import torch
import torch.nn as nn
import torch.nn.functional as F

from ._3dmaxfilter import MaxFiltering

def xcorr_depthwise(x, kernel):
    """depthwise cross correlation
    """
    batch = kernel.size(0)  # batch
    channel = kernel.size(1)  # 256
    x = x.view(1, batch*channel, x.size(2), x.size(3))
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch*channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out

class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

class BAN(nn.Module):
    def __init__(self):
        super(BAN, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError


class DepthwiseXCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3):
        super(DepthwiseXCorr, self).__init__()
        self.conv_kernel = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.conv_search = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1)
        )

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel)  # (28,256, 5, 5)
        search = self.conv_search(search)  # (28, 256, 29,29)
        feature = xcorr_depthwise(search, kernel)  # (28, 256, 25, 25)
        out = self.head(feature)  # (28, 2, 25, 25)
        return out, feature

class DepthwiseBAN(BAN):
    def __init__(self, in_channels=256, out_channels=256, cls_out_channels=2, weighted=False):
        super(DepthwiseBAN, self).__init__()
        self.cls = DepthwiseXCorr(in_channels, out_channels, cls_out_channels)  # DepthwiseXCorr 过程
        self.loc = DepthwiseXCorr(in_channels, out_channels, 4)

    def forward(self, z_f, x_f):
        cls, _ = self.cls(z_f, x_f)  # (28,2,25,25)
        loc, loc_feature = self.loc(z_f, x_f)  # (28,4,25,25)
        return cls, loc, loc_feature

class MultiBAN(BAN):
    def __init__(self, in_channels, cls_out_channels, weighted=False, need_weighted=None):  #  原:in_channels : [256,256,256] \\ cls_out_channels:2 \\weighted: True
        super(MultiBAN, self).__init__()
        self.weighted = weighted  # True
        self.fpn_strides = [8, 8, 8, 16, 32]
        for i in range(len(in_channels)):
            self.add_module('box'+str(i+3), DepthwiseBAN(in_channels[i], in_channels[i], cls_out_channels))
        if self.weighted:
            self.cls_weight = nn.Parameter(torch.ones(len(need_weighted)))  # 3个head输出占的比重
            self.loc_weight = nn.Parameter(torch.ones(len(need_weighted)))  # 这里也要改，
        # self.loc_scale = nn.Parameter(torch.ones(len(in_channels)))

        self.max3d = MaxFiltering(in_channels[0],
        kernel_size=3,
        tau=2)
        num_shifts = 1
        self.filter = nn.Conv2d(in_channels[0],
                                num_shifts,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.loc_scale = nn.ModuleList(
            [Scale(init_value=1.0) for _ in range(len(self.fpn_strides))])  # 用于将其乘以回归头

    def forward(self, z_fs, x_fs):

        def weighted_avg(lst, weight):  # 每个输出乘以权重再加和，这个权重已经softmax过了，所以加和后并没有进行平均操作。
            s = 0
            for i in range(len(weight)):
                s += lst[i] * weight[i]
            return s
        cls = []
        loc = []
        cls_first_three = []
        loc_first_three = []
        filter_first_three = []
        filter_subnet = []
        for idx, (z_f, x_f) in enumerate(zip(z_fs, x_fs), start=3):  #  按 fpn level 拿出每层的z 和 x 的特征
            box = getattr(self, 'box'+str(idx))  # 由 DepthwiseBAN 构成
            c, l, loc_feature = box(z_f, x_f)
            if idx-3 < 3:
                filter_first_three.append(loc_feature)
                cls_first_three.append(c)
                l = self.loc_scale[idx-3](l)
                loc_first_three.append(F.relu(l) * self.fpn_strides[idx-3])
                if idx-3 == 2:
                    cls_weight = F.softmax(self.cls_weight, 0)
                    loc_weight = F.softmax(self.loc_weight, 0)
                    cls.append(weighted_avg(cls_first_three, cls_weight))
                    loc.append(weighted_avg(loc_first_three, loc_weight))
                    filter_subnet.append(weighted_avg(filter_first_three, loc_weight))
            else:
                filter_subnet.append(loc_feature)
                cls.append(c)
                l = self.loc_scale[idx-3](l)
                # loc.append(torch.exp(l*self.loc_scale[idx-3]))
                loc.append(F.relu(l) * self.fpn_strides[idx-3])
        filters = [self.filter(x) for x in self.max3d(filter_subnet)]

        return cls, loc, filters

        # if self.weighted:
        #     cls_weight = F.softmax(self.cls_weight, 0)
        #     loc_weight = F.softmax(self.loc_weight, 0)
        #     # cls_weight_numpy = cls_weight.detach().cpu().numpy()
        #     # loc_weight_numpy = loc_weight.detach().cpu().numpy()
        #     # print("1")
        #
        # def avg(lst):
        #     return sum(lst) / len(lst)
        #
        # def weighted_avg(lst, weight):  # 每个输出乘以权重再加和，这个权重已经softmax过了，所以加和后并没有进行平均操作。
        #     s = 0
        #     for i in range(len(weight)):
        #         s += lst[i] * weight[i]
        #     return s
        # #
        # if self.weighted:
        #     return weighted_avg(cls, cls_weight), weighted_avg(loc, loc_weight)  # 返回乘以权重然后再加和的输出
        # else:
        #     return avg(cls), avg(loc)

if __name__ == '__main__':
    need_weight = ['p3', 'p4', 'p5']
    in_channel = [256, 256, 256, 256, 256]
    cls_out_channel = 2
    head = MultiBAN(in_channel, cls_out_channel, True, need_weight)
    print(head)