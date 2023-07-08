from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from siamban.core.xcorr import xcorr_fast, xcorr_depthwise

class BAN(nn.Module):
    def __init__(self):
        super(BAN, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError

class UPChannelBAN(BAN):
    def __init__(self, feature_in=256, cls_out_channels=2):
        super(UPChannelBAN, self).__init__()

        cls_output = cls_out_channels
        loc_output = 4

        self.template_cls_conv = nn.Conv2d(feature_in, 
                feature_in * cls_output, kernel_size=3)
        self.template_loc_conv = nn.Conv2d(feature_in, 
                feature_in * loc_output, kernel_size=3)

        self.search_cls_conv = nn.Conv2d(feature_in, 
                feature_in, kernel_size=3)
        self.search_loc_conv = nn.Conv2d(feature_in, 
                feature_in, kernel_size=3)

        self.loc_adjust = nn.Conv2d(loc_output, loc_output, kernel_size=1)


    def forward(self, z_f, x_f):
        cls_kernel = self.template_cls_conv(z_f)
        loc_kernel = self.template_loc_conv(z_f)

        cls_feature = self.search_cls_conv(x_f)
        loc_feature = self.search_loc_conv(x_f)

        cls = xcorr_fast(cls_feature, cls_kernel)
        loc = self.loc_adjust(xcorr_fast(loc_feature, loc_kernel))
        return cls, loc


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
        return out


class DepthwiseBAN(BAN):
    def __init__(self, in_channels=256, out_channels=256, cls_out_channels=2, weighted=False):
        super(DepthwiseBAN, self).__init__()
        self.cls = DepthwiseXCorr(in_channels, out_channels, cls_out_channels)  # DepthwiseXCorr 过程
        self.loc = DepthwiseXCorr(in_channels, out_channels, 4)

    def forward(self, z_f, x_f):
        cls = self.cls(z_f, x_f)  # (28,2,25,25)
        loc = self.loc(z_f, x_f)  # (28,4,25,25)
        return cls, loc


class MultiBAN(BAN):
    def __init__(self, in_channels, cls_out_channels, weighted=False):
        super(MultiBAN, self).__init__()
        self.weighted = weighted  # True
        for i in range(len(in_channels)):
            self.add_module('box'+str(i+2), DepthwiseBAN(in_channels[i], in_channels[i], cls_out_channels))
        if self.weighted:
            self.cls_weight = nn.Parameter(torch.ones(len(in_channels)))  # 3个head输出占的比重
            self.loc_weight = nn.Parameter(torch.ones(len(in_channels)))
        self.loc_scale = nn.Parameter(torch.ones(len(in_channels)))

    def forward(self, z_fs, x_fs):
        cls = []
        loc = []
        for idx, (z_f, x_f) in enumerate(zip(z_fs, x_fs), start=2):  # 索引从2开始
            box = getattr(self, 'box'+str(idx))  # 由 DepthwiseBAN 构成
            c, l = box(z_f, x_f)
            cls.append(c)
            loc.append(torch.exp(l*self.loc_scale[idx-2]))

        if self.weighted:
            cls_weight = F.softmax(self.cls_weight, 0)
            loc_weight = F.softmax(self.loc_weight, 0)
            # cls_weight_numpy = cls_weight.detach().cpu().numpy()
            # loc_weight_numpy = loc_weight.detach().cpu().numpy()
            # print("1")

        def avg(lst):
            return sum(lst) / len(lst)

        def weighted_avg(lst, weight):  # 每个输出乘以权重再加和，这个权重已经softmax过了，所以加和后并没有进行平均操作。
            s = 0
            for i in range(len(weight)):
                s += lst[i] * weight[i]
            return s

        if self.weighted:
            return weighted_avg(cls, cls_weight), weighted_avg(loc, loc_weight)  # 返回乘以权重然后再加和的输出
        else:
            return avg(cls), avg(loc)
