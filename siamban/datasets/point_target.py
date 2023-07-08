from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch

from siamban.core.config import cfg
from siamban.utils.bbox import corner2center
from siamban.utils.point import Point


class PointTarget:
    def __init__(self,):
        self.points = Point(cfg.POINT.STRIDE, cfg.POINT.TRAIN_OUTPUT_SIZE, cfg.TRAIN.SEARCH_SIZE//2)  # {PointTarget} points={ndarray:(2,25,25)} <c>

    def __call__(self, target, size):  # size：25 target:bbox-->{Corner:4} 传入的是bbox,  cfg.TRAIN.OUTPUT_SIZE, neg
        tcx, tcy, tw, th = corner2center(target)
        points = self.points.points

        # def select(position, keep_num=16):  # keep_num 16 或 48
        #     num = position[0].shape[0]  # 举例 569
        #     if num <= keep_num:
        #         return position, num
        #     slt = np.arange(num)  # 举例 ndarray:(569:)  [0~568]
        #     np.random.shuffle(slt)  # 打乱 随机选取
        #     slt = slt[:keep_num]  # ndarray:(48,)
        #     return tuple(p[slt] for p in position), keep_num  # position  <c> 选中对应slt的 position中的横纵坐标

        # -1 ignore 0 negative 1 positive
        cls_label = []
        delta_label = []
        for i in range(len(size)):
            cls = np.zeros((1, size[i], size[i]), dtype=np.int64)  # ndarray:(25,25)  <c>
            delta = np.zeros((4, size[i], size[i]), dtype=np.float32)  # ndarray:(4,25,25) <c>
            delta[0] = points[i][0] - target[0]
            delta[1] = points[i][1] - target[1]
            delta[2] = target[2] - points[i][0]
            delta[3] = target[3] - points[i][1]
            # pos = np.where(np.square(tcx - points[i][0]) / np.square(tw / 4) +
            #                np.square(tcy - points[i][1]) / np.square(th / 4) < 1)  # <c> tuple:2  对应论文中的E2部分
            pos = np.where(np.square(tcx - points[i][0]) / np.square(tw / 2) +
                           np.square(tcy - points[i][1]) / np.square(th / 2) < 1)
            # neg = np.where(np.square(tcx - points[i][0]) / np.square(tw / 2) +
            #                np.square(tcy - points[i][1]) / np.square(th / 2) >= 1)
            cls[0][pos] = 1
            # cls[1][neg] = 1
            # ==  这里为了做实验而故意设计的多增加一维，实际这里不需要这样 ===================
            # cls = torch.from_numpy(cls)
            # delta = torch.from_numpy(delta)
            # cls = cls.unsqueeze(0)
            # delta = delta.unsqueeze(0)
            # cls_label.append(cls)
            # delta_label.append(delta)
            # ========== 下面的为实际的 ==============================================
            # cls = torch.from_numpy(cls)
            cls_label.append(torch.from_numpy(cls))
            delta_label.append(torch.from_numpy(delta))
        return cls_label, delta_label
