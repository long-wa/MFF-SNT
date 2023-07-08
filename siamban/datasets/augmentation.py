# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import cv2

from siamban.utils.bbox import corner2center, \
        Center, center2corner, Corner


class Augmentation:
    def __init__(self, shift, scale, blur, flip, color):
        self.shift = shift
        self.scale = scale
        self.blur = blur
        self.flip = flip
        self.color = color
        self.rgbVar = np.array(
            [[-0.55919361,  0.98062831, - 0.41940627],
             [1.72091413,  0.19879334, - 1.82968581],
             [4.64467907,  4.73710203, 4.88324118]], dtype=np.float32)

    @staticmethod
    def random():
        return np.random.random() * 2 - 1.0

    def _crop_roi(self, image, bbox, out_sz, padding=(0, 0, 0)):
        bbox = [float(x) for x in bbox]  # list:4 [, , ,]
        a = (out_sz-1) / (bbox[2]-bbox[0])
        b = (out_sz-1) / (bbox[3]-bbox[1])
        c = -a * bbox[0]
        d = -b * bbox[1]
        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float)
        crop = cv2.warpAffine(image, mapping, (out_sz, out_sz),
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=padding)
        return crop

    def _blur_aug(self, image):
        def rand_kernel():
            sizes = np.arange(5, 46, 2)
            size = np.random.choice(sizes)
            kernel = np.zeros((size, size))
            c = int(size/2)
            wx = np.random.random()
            kernel[:, c] += 1. / size * wx
            kernel[c, :] += 1. / size * (1-wx)
            return kernel
        kernel = rand_kernel()
        image = cv2.filter2D(image, -1, kernel)
        return image

    def _color_aug(self, image):
        offset = np.dot(self.rgbVar, np.random.randn(3, 1))
        offset = offset[::-1]  # bgr 2 rgb
        offset = offset.reshape(3)
        image = image - offset
        return image

    def _gray_aug(self, image):
        grayed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(grayed, cv2.COLOR_GRAY2BGR)
        return image

    def _shift_scale_aug(self, image, bbox, crop_bbox, size):
        im_h, im_w = image.shape[:2]  # 511,511

        # adjust crop bounding box
        crop_bbox_center = corner2center(crop_bbox)  # {Center:4} <c>
        if self.scale:  # True
            scale_x = (1.0 + Augmentation.random() * self.scale)  # 举例 1.106232342342423  ration 尺度变换
            scale_y = (1.0 + Augmentation.random() * self.scale)  # 举例 0.9921912342141
            h, w = crop_bbox_center.h, crop_bbox_center.w  # 举例 h:254.0 w:254.0
            scale_x = min(scale_x, float(im_w) / w)  # 数值  保留扩大的
            scale_y = min(scale_y, float(im_h) / h)  # 数值
            crop_bbox_center = Center(crop_bbox_center.x,
                                      crop_bbox_center.y,
                                      crop_bbox_center.w * scale_x,
                                      crop_bbox_center.h * scale_y)  # <c>  以目标为中心，对bbox 的 w h 尺度进行相应的扩大

        crop_bbox = center2corner(crop_bbox_center)  # <c>  转成角的形式  尺度变换后的
        if self.shift:  # True
            sx = Augmentation.random() * self.shift  # 举例 -10.489860021231231  在 x和y方向随机设置偏移量
            sy = Augmentation.random() * self.shift  # 举例 26.7879797979789

            x1, y1, x2, y2 = crop_bbox

            sx = max(-x1, min(im_w - 1 - x2, sx))  # 数值  若偏移后还在图片范围内（511，511），则保留偏移量，超出范围则舍掉
            sy = max(-y1, min(im_h - 1 - y2, sy))  # 数值

            crop_bbox = Corner(x1 + sx, y1 + sy, x2 + sx, y2 + sy)  # 添加偏移量，作为裁剪基准的矩形框,只在x和y方向综合移动，并不改变矩形框的大小，上面的尺度变换会改变矩形框的大小

        # adjust target bounding box
        x1, y1 = crop_bbox.x1, crop_bbox.y1
        bbox = Corner(bbox.x1 - x1, bbox.y1 - y1,
                      bbox.x2 - x1, bbox.y2 - y1)

        if self.scale:
            bbox = Corner(bbox.x1 / scale_x, bbox.y1 / scale_y,
                          bbox.x2 / scale_x, bbox.y2 / scale_y)

        image = self._crop_roi(image, crop_bbox, size)  # 裁剪
        return image, bbox

    def _flip_aug(self, image, bbox):
        image = cv2.flip(image, 1)
        width = image.shape[1]
        bbox = Corner(width - 1 - bbox.x2, bbox.y1,
                      width - 1 - bbox.x1, bbox.y2)
        return image, bbox

    def __call__(self, image, bbox, size, gray=False):  # size:   127 或 255
        shape = image.shape  # (511,511,3)
        crop_bbox = center2corner(Center(shape[0]//2, shape[1]//2,
                                         size-1, size-1))  # {Corner:4} <c> 以图片为中心，尺寸为 size-1
        # gray augmentation
        if gray:  # False
            image = self._gray_aug(image)

        # shift scale augmentation
        image, bbox = self._shift_scale_aug(image, bbox, crop_bbox, size)  # image:(255,255,3) bbox {Corner:4} <c>

        # color augmentation
        if self.color > np.random.random():  # True
            image = self._color_aug(image)

        # blur augmentation
        if self.blur > np.random.random():  # False
            image = self._blur_aug(image)

        # flip augmentation
        if self.flip and self.flip > np.random.random():  # False
            image, bbox = self._flip_aug(image, bbox)
        return image, bbox
