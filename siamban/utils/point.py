from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np


class Point:
    """
    This class generate points.
    """
    def __init__(self, stride, size, image_center):
        self.stride = stride  # [8,16,32]
        self.size = size  # [17,9,5]
        self.points =[]
        self.image_center = image_center  # 127 这个是由 search_size / 2 得来的，所以这个点的坐标是映射到搜索区域的
        for _, (stride_l, size_l) in enumerate(zip(self.stride, self.size)):
            self.points.append(self.generate_points(stride_l, size_l, self.image_center))
        # self.points = self.generate_points(self.stride, self.size, self.image_center)

    def generate_points(self, stride, size, im_c):
        ori = im_c - size // 2 * stride  # 31 把输出得分图的中心和图片的中心对齐，以这两点作为参考点求其差值
        x, y = np.meshgrid([ori + stride * dx for dx in np.arange(0, size)],
                           [ori + stride * dy for dy in np.arange(0, size)])  # 返回网格坐标点x, y，两个同型矩阵，两个矩阵对应位置的数的组合即为 输出特征图上每个位置的坐标
        points = np.zeros((2, size, size), dtype=np.float32)
        points[0, :, :], points[1, :, :] = x.astype(np.float32), y.astype(np.float32)  # 第一维x，第二维是y

        return points  # 得分图上的点返回原图像上的坐标， ndarray:(2,25,25),第一维是x的坐标，第二维是y的坐标

if __name__ == "__main__":
    point = Point(8, 25, 255 // 2)
    points1 = point.points
    print(points1)