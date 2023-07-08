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
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.POINT.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        # hanning = np.hanning(self.score_size)
        # window = np.outer(hanning, hanning)

        self.name = "siamBAN"
        self.cls_out_channels = cfg.BAN.KWARGS.cls_out_channels
        # self.window = window.flatten()
        self.points = self.generate_points(cfg.POINT.STRIDE, self.score_size)
        self.model = model
        self.model.eval()

    def generate_points(self, stride, size):
        ori = - (size // 2) * stride
        x, y = np.meshgrid([ori + stride * dx for dx in np.arange(0, size)],
                           [ori + stride * dy for dy in np.arange(0, size)])
        points = np.zeros((size * size, 2), dtype=np.float32)
        points[:, 0], points[:, 1] = x.astype(np.float32).flatten(), y.astype(np.float32).flatten()

        return points

    def _convert_bbox(self, delta, point):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.detach().cpu().numpy()

        delta[0, :] = point[:, 0] - delta[0, :]
        delta[1, :] = point[:, 1] - delta[1, :]
        delta[2, :] = point[:, 0] + delta[2, :]
        delta[3, :] = point[:, 1] + delta[3, :]
        delta[0, :], delta[1, :], delta[2, :], delta[3, :] = corner2center(delta)
        return delta

    def _convert_score(self, score):
        if self.cls_out_channels == 1:
            score = score.permute(1, 2, 3, 0).contiguous().view(-1)
            score = score.sigmoid().detach().cpu().numpy()
        else:
            score = score.permute(1, 2, 3, 0).contiguous().view(self.cls_out_channels, -1).permute(1, 0)
            score = score.softmax(1).detach()[:, 1].cpu().numpy()
        return score

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
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        self.model.template(z_crop)

    # def track(self, img):
    #     """
    #     args:
    #         img(np.ndarray): BGR image
    #     return:
    #         bbox(list):[x, y, width, height]
    #     """
    #     w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
    #     h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
    #     s_z = np.sqrt(w_z * h_z)
    #     scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
    #     s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
    #     # -------------------------- 更改 处 -----------------------------------
    #     x_crop_new, instance_size_new, response_size_new = search_area_change(self.size, img, s_x,
    #                                                                           response_size_ori=cfg.TRAIN.OUTPUT_SIZE)
    #
    #     print("response_size_new of shape is {}".format(response_size_new))
    #     if response_size_new==13:
    #         print("here")
    #
    #     x_crop = self.get_subwindow(img, self.center_pos,
    #                                 instance_size_new,
    #                                 x_crop_new, self.channel_average)
    #
    #     outputs = self.model.track(x_crop)
    #     # ------------------------------- point 筛选区域 ------------------------------
    #     index_points_begin = int((cfg.TRAIN.OUTPUT_SIZE + 1) / 2 - response_size_new // 2)
    #     index_points_terminal = int((cfg.TRAIN.OUTPUT_SIZE + 1) / 2 + (response_size_new + 1) / 2)
    #     points_channel1 = self.points[:, 0].reshape(cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.OUTPUT_SIZE)
    #     points_channel2 = self.points[:, 1].reshape(cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.OUTPUT_SIZE)
    #     select_points_1 = points_channel1[index_points_begin:index_points_terminal, index_points_begin:index_points_terminal]
    #     select_points_2 = points_channel2[index_points_begin:index_points_terminal, index_points_begin:index_points_terminal]
    #     select_points = np.zeros((response_size_new * response_size_new, 2))
    #     select_points[:, 0], select_points[:, 1] = select_points_1.flatten(), select_points_2.flatten()
    #
    #
    #     score = self._convert_score(outputs['cls'])
    #     pred_bbox = self._convert_bbox(outputs['loc'], select_points)
    #
    #     def change(r):
    #         return np.maximum(r, 1. / r)
    #
    #     def sz(w, h):
    #         pad = (w + h) * 0.5
    #         return np.sqrt((w + pad) * (h + pad))
    #
    #     # scale penalty
    #     s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
    #                  (sz(self.size[0]*scale_z, self.size[1]*scale_z)))
    #
    #     # aspect ratio penalty
    #     r_c = change((self.size[0]/self.size[1]) /
    #                  (pred_bbox[2, :]/pred_bbox[3, :]))
    #     penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
    #     pscore = penalty * score
    #
    #     # window penalty
    #     hanning = np.hanning(response_size_new)  # 与下面搭配使用，生成高斯函数
    #     window = np.outer(hanning, hanning)
    #     pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
    #         window.flatten() * cfg.TRACK.WINDOW_INFLUENCE
    #     best_idx = np.argmax(pscore)
    #     bbox = pred_bbox[:, best_idx] / scale_z
    #     lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR
    #
    #     cx = bbox[0] + self.center_pos[0]
    #     cy = bbox[1] + self.center_pos[1]
    #
    #     # smooth bbox
    #     width = self.size[0] * (1 - lr) + bbox[2] * lr
    #     height = self.size[1] * (1 - lr) + bbox[3] * lr
    #
    #     # clip boundary
    #     cx, cy, width, height = self._bbox_clip(cx, cy, width,
    #                                             height, img.shape[:2])
    #
    #     # udpate state
    #     self.center_pos = np.array([cx, cy])
    #     self.size = np.array([width, height])
    #
    #     bbox = [cx - width / 2,
    #             cy - height / 2,
    #             width,
    #             height]
    #     best_score = score[best_idx]
    #     return {
    #             'bbox': bbox,
    #             'best_score': best_score
    #            }


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def update(self, img):
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
        # -------------------------- 更改 处 -----------------------------------
        x_crop_new, instance_size_new, response_size_new = search_area_change(self.size, img, s_x,
                                                                              response_size_ori=cfg.TRAIN.OUTPUT_SIZE)
        print("response_size_new of shape is {}".format(response_size_new))

        x_crop = self.get_subwindow(img, self.center_pos,
                                    instance_size_new,
                                    x_crop_new, self.channel_average)

        outputs = self.model.track(x_crop)
        # ------------------------------- point 筛选区域 ------------------------------
        index_points_begin = int((cfg.TRAIN.OUTPUT_SIZE + 1) / 2 - response_size_new // 2)
        index_points_terminal = int((cfg.TRAIN.OUTPUT_SIZE + 1) / 2 + (response_size_new + 1) / 2)
        points_channel1 = self.points[:, 0].reshape(cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.OUTPUT_SIZE)
        points_channel2 = self.points[:, 1].reshape(cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.OUTPUT_SIZE)
        select_points_1 = points_channel1[index_points_begin:index_points_terminal, index_points_begin:index_points_terminal]
        select_points_2 = points_channel2[index_points_begin:index_points_terminal, index_points_begin:index_points_terminal]
        select_points = np.zeros((response_size_new * response_size_new, 2))
        select_points[:, 0], select_points[:, 1] = select_points_1.flatten(), select_points_2.flatten()

        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], select_points)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0] * scale_z, self.size[1] * scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0] / self.size[1]) /
                     (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        hanning = np.hanning(response_size_new)  # 与下面搭配使用，生成高斯函数
        window = np.outer(hanning, hanning)
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                 window.flatten() * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)
        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

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
        return bbox

    def track(self, img_files, box, visualize=False):
        frame_num = len(img_files)  # 帧数
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box  # boxes中第一个就是box
        times = np.zeros(frame_num)

        def read_image(img_file, cvt_code=cv2.COLOR_BGR2RGB):  # 读入img
            img = cv2.imread(img_file, cv2.IMREAD_COLOR)
            if cvt_code is not None:
                img = cv2.cvtColor(img, cvt_code)  # 转为灰度图
            return img

        for f, img_file in enumerate(img_files):
            img = read_image(img_file)  # 读入img

            begin = time.time()  # 计时
            if f == 0:
                self.init(img, box)
            else:
                boxes[f, :] = self.update(img)  # 更新box
            times[f] = time.time() - begin

            if visualize:
                # ops.show_image(img, boxes[f, :])
                print('None')
            print('{}/{}'.format(f, frame_num))
            sys.stdout.flush()

        return boxes, times

