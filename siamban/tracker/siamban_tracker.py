from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np
import torch
import time
import sys

from siamban.core.config import cfg
from siamban.tracker.base_tracker import SiameseTracker
from siamban.utils.bbox import corner2center
from my_experiments.search_area_change_strategy import search_area_change

class SiamBANTracker(SiameseTracker):
    def __init__(self, model):
        super(SiamBANTracker, self).__init__()
        # self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
        #     cfg.POINT.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        # hanning = np.hanning(self.score_size)
        # window = np.outer(hanning, hanning)
        # 名字
        self.name = "Ours"
        self.is_deterministic = False
        self.cls_out_channels = cfg.BAN.KWARGS.cls_out_channels  # 2
        # self.window = window.flatten()
        self.points = self.generate_points(cfg.POINT.STRIDE, cfg.POINT.TRAIN_OUTPUT_SIZE)
        self.model = model
        self.model.eval()

        # =======
        self.window = self.get_window(cfg.POINT.TRAIN_OUTPUT_SIZE)  # (395,)
        self.idx = 1

    def generate_points(self, stride, size):
        points_list = []
        for _, (stride_l, size_l) in enumerate(zip(stride, size)):
            ori = - (size_l // 2) * stride_l
            x, y = np.meshgrid([ori + stride_l * dx for dx in np.arange(0, size_l)],
                               [ori + stride_l * dy for dy in np.arange(0, size_l)])
            points = np.zeros((size_l * size_l, 2), dtype=np.float32)
            points[:, 0], points[:, 1] = x.astype(np.float32).flatten(), y.astype(np.float32).flatten()
            points_list.append(points)

        return points_list

    def _convert_bbox(self, delta, point):
        delta_pre = None
        for _, (delta_l, point_l) in enumerate(zip(delta, point)):  # delta_l Tensor:(1,5,17,17), point_l: ndarray:(289,2)

            delta_l = delta_l.permute(1, 2, 3, 0).contiguous().view(4, -1)
            delta_l = delta_l.detach().cpu().numpy()  # (4, 289)

            delta_l[0, :] = point_l[:, 0] - delta_l[0, :]
            delta_l[1, :] = point_l[:, 1] - delta_l[1, :]
            delta_l[2, :] = point_l[:, 0] + delta_l[2, :]
            delta_l[3, :] = point_l[:, 1] + delta_l[3, :]
            delta_l[0, :], delta_l[1, :], delta_l[2, :], delta_l[3, :] = corner2center(delta_l)
            if _ == 0:
                delta_pre = delta_l
            else:
                delta_pre = np.hstack((delta_pre, delta_l))
        return delta_pre

    def _convert_score(self, score, filters):
        if self.cls_out_channels == 1:
            score = score.permute(1, 2, 3, 0).contiguous().view(-1)
            score = score.sigmoid().detach().cpu().numpy()
        else:
            score = score.permute(1, 2, 3, 0).contiguous().view(self.cls_out_channels, -1).permute(1, 0)
            score = score.softmax(1).detach()[:, 1].cpu().numpy()
        return score

    def get_window(self, size_list):
        windows = None
        for i in range(len(size_list)):
            hanning = np.hanning(size_list[i])
            window = np.outer(hanning, hanning)
            window = window.flatten()
            if i == 0:
                windows = window
            else:
                windows=np.hstack((windows, window))
        return windows



    def cat(self, tensors, dim=0):
        """
        Efficient version of torch.cat that avoids a copy if there is only a single element in a list
        """
        assert isinstance(tensors, (list, tuple))
        if len(tensors) == 1:
            return tensors[0]
        return torch.cat(tensors, dim)

    def permute_to_N_HWA_K(self, tensor, K):
        """
        Transpose/reshape a tensor from (N, (A x K), H, W) to (N, (HxWxA), K)
        """
        assert tensor.dim() == 4, tensor.shape
        N, _, H, W = tensor.shape
        tensor = tensor.view(N, -1, K, H, W)
        tensor = tensor.permute(0, 3, 4, 1, 2)
        tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)  举例 : Tensor:(2,10752,80)
        return tensor

    def permute_all_cls_and_box_to_N_HWA_K_and_concat(
            self, box_cls,  box_center, num_classes=2
    ):
        box_cls_flattened = [self.permute_to_N_HWA_K(x, num_classes) for x in box_cls]
        box_cls = self.cat(box_cls_flattened, dim=1).view(-1, num_classes)
        box_center_flattened = [self.permute_to_N_HWA_K(x, 1) for x in box_center]
        box_center = self.cat(box_center_flattened, dim=1).view(-1, 1)
        return box_cls, box_center


    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    @torch.no_grad()
    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop, z_crop_img = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        # cv2.imwrite('D:/subject2/crop_img/z_01.jpg', z_crop_img)
        self.model.template(z_crop)

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop, x_crop_img = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)
        # cv2.imwrite('D:/subject2/crop_img/x_{}.jpg'.format(self.idx), x_crop_img)
        # self.idx += 1

        outputs = self.model.track(x_crop)
        cls, filters = self.permute_all_cls_and_box_to_N_HWA_K_and_concat(outputs['cls'], outputs['filters'])
        score = cls.sigmoid() * filters.sigmoid()  # (395,2)
        score = score.softmax(1).detach()[:, 0].cpu().numpy()  # ndarray:(395,)

        # score = self._convert_score(outputs['cls'], outputs['filters'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.points)  # (4,395)

        # def change(r):
        #     return np.maximum(r, 1. / r)
        #
        # def sz(w, h):
        #     pad = (w + h) * 0.5
        #     return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        # s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
        #              (sz(self.size[0]*scale_z, self.size[1]*scale_z)))
        #
        # # aspect ratio penalty
        # r_c = change((self.size[0]/self.size[1]) /
        #              (pred_bbox[2, :]/pred_bbox[3, :]))
        # penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        # pscore = penalty * score

        # window penalty
        score = score * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(score)
        bbox = pred_bbox[:, best_idx] / scale_z
        # lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = bbox[2]
        height = bbox[3]

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]
        return {
                'bbox': bbox,
                'best_score': best_score
               }


# # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#     @torch.no_grad()
#     def update(self, img):
#         w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
#         h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
#         s_z = np.sqrt(w_z * h_z)
#         scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
#         s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
#         x_crop, _ = self.get_subwindow(img, self.center_pos,
#                                     cfg.TRACK.INSTANCE_SIZE,
#                                     round(s_x), self.channel_average)
#
#         outputs = self.model.track(x_crop)
#         cls, filters = self.permute_all_cls_and_box_to_N_HWA_K_and_concat(outputs['cls'], outputs['filters'])
#         score = cls.sigmoid() * filters.sigmoid()  # (395,2)
#         score = score.softmax(1).detach()[:, 0].cpu().numpy()  # ndarray:(395,)
#
#         # score = self._convert_score(outputs['cls'], outputs['filters'])
#         pred_bbox = self._convert_bbox(outputs['loc'], self.points)  # (4,395)
#
#         # def change(r):
#         #     return np.maximum(r, 1. / r)
#         #
#         # def sz(w, h):
#         #     pad = (w + h) * 0.5
#         #     return np.sqrt((w + pad) * (h + pad))
#
#         # scale penalty
#         # s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
#         #              (sz(self.size[0]*scale_z, self.size[1]*scale_z)))
#         #
#         # # aspect ratio penalty
#         # r_c = change((self.size[0]/self.size[1]) /
#         #              (pred_bbox[2, :]/pred_bbox[3, :]))
#         # penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
#         # pscore = penalty * score
#
#         # window penalty
#         # score = score * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
#         #     self.window * cfg.TRACK.WINDOW_INFLUENCE
#         best_idx = np.argmax(score)
#         bbox = pred_bbox[:, best_idx] / scale_z
#         # lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR
#
#         cx = bbox[0] + self.center_pos[0]
#         cy = bbox[1] + self.center_pos[1]
#
#         # smooth bbox
#         width = bbox[2]
#         height = bbox[3]
#
#         # clip boundary
#         cx, cy, width, height = self._bbox_clip(cx, cy, width,
#                                                 height, img.shape[:2])
#
#         # udpate state
#         self.center_pos = np.array([cx, cy])
#         self.size = np.array([width, height])
#
#         bbox = [cx - width / 2,
#                 cy - height / 2,
#                 width,
#                 height]
#         best_score = score[best_idx]
#         return bbox
#
#     def track(self, img_files, box, visualize=False):
#         frame_num = len(img_files)  # 帧数
#         boxes = np.zeros((frame_num, 4))
#         boxes[0] = box  # boxes中第一个就是box
#         times = np.zeros(frame_num)
#
#         def read_image(img_file, cvt_code=cv2.COLOR_BGR2RGB):  # 读入img
#             img = cv2.imread(img_file, cv2.IMREAD_COLOR)
#             if cvt_code is not None:
#                 img = cv2.cvtColor(img, cvt_code)  # 转为灰度图
#             return img
#
#         for f, img_file in enumerate(img_files):
#             img = read_image(img_file)  # 读入img
#
#             begin = time.time()  # 计时
#             if f == 0:
#                 self.init(img, box)
#             else:
#                 boxes[f, :] = self.update(img)  # 更新box
#             times[f] = time.time() - begin
#
#             if visualize:
#                 # ops.show_image(img, boxes[f, :])
#                 print('None')
#             print('{}/{}'.format(f, frame_num))
#             sys.stdout.flush()
#
#         return boxes, times


