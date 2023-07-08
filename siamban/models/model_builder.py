# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from siamban.core.config import cfg
from siamban.models.loss import select_cross_entropy_loss, select_iou_loss, focal_loss
from siamban.models.backbone import get_backbone
# from siamban.models.head import get_ban_head
# from siamban.models.neck import get_neck

# =================== 修改的 =====================
from siamban.models.neck.neck import FPN, conv_with_kaiming_uniform, LastLevelP6P7
from siamban.models.head_create import MultiBAN
from matplotlib import pyplot as plt


def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)

def permute_to_N_HWA_K(tensor, K):
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
    box_cls, box_delta, box_center, num_classes=2
):
    box_cls_flattened = [permute_to_N_HWA_K(x, num_classes) for x in box_cls]
    box_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in box_delta]
    box_cls = cat(box_cls_flattened, dim=1).view(-1, num_classes)
    box_delta = cat(box_delta_flattened, dim=1).view(-1, 4)
    if box_center is not None:
        box_center_flattened = [permute_to_N_HWA_K(x, 1) for x in box_center]
        box_center = cat(box_center_flattened, dim=1).view(-1, 1)
        return box_cls, box_delta, box_center
    else:
        return box_cls, box_delta

# def show_feature_map(feature_list, name, index, each_row=4):
#     for i, feature_map in enumerate(feature_list):
#         feature_map = feature_map.cpu().detach().numpy().squeeze()
#         for j in range(4):
#             feature = feature_map[j*each_row]
#             feature = (feature - feature.min()) / (feature.max() - feature.min())
#             for k in range(1, each_row):
#                 tem_feature = feature_map[j*each_row + k]
#                 tem_feature = (tem_feature - tem_feature.min()) / (tem_feature.max() - tem_feature.min())
#                 feature = np.hstack((feature, tem_feature))
#             if j == 0:
#                 ans =feature
#             else:
#                 ans = np.vstack((ans, feature))
#         plt.figure(figsize=(15, 15))
#         plt.imshow(ans, cmap='gray')
#         plt.savefig('D:/subject2/featuremap_visualize/{}_{}_layer{}.jpg'.format(name, index, i))
#         # feature_ave = np.sum(feature_map, axis=0) / feature_map.shape[0]
#         # feature_ave = feature_map[0]
#         # feature_ave = (feature_ave - feature_ave.min()) / (feature_ave.max() - feature_ave.min())
#         # feature = np.asarray(feature_ave*255, dtype=np.uint8)
#         # feature = cv2.resize(feature, (224, 224), interpolation=cv2.INTER_NEAREST)
#         # feature = cv2.applyColorMap(feature, cv2.COLORMAP_JET)
#         # cv2.imwrite('D:/subject2/featuremap_visualize/{}_{}_layer{}.jpg'.format(name, index, i), feature)




class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)  # 搭建主干网络

        self.fpn = FPN(
            in_channels_list=cfg.FPN.IN_CHANNELS,
            out_channels=cfg.FPN.OUT_CHANNELS,
            conv_block=conv_with_kaiming_uniform(cfg.FPN.USE_GN, cfg.FPN.USE_RELU),
            top_blocks=LastLevelP6P7(cfg.FPN.IN_CHANNELS_P6P7, cfg.FPN.OUT_CHANNELS)
        )

        self.ban = MultiBAN(cfg.HEAD.IN_CHANNEL, cfg.HEAD.CLS_OUT_CHANNEL, True,
                             need_weighted=cfg.HEAD.NEED_WEIGHT)

        # =======================
        # self.index = 1

    def template(self, z):
        zf = self.backbone(z)  # {list:3} [Tensor:(1,512,15,15),Tensor:(1,1024,15,15),Tensor:(1,2048,15,15)]
        # l = 4
        # r = l + 7
        # for i in range(len(zf)):
        #     zf[i] = zf[i][:, :, l:r, l:r]
        zf = self.fpn(zf)  # tuple:5 {Tensor:(1,256,15,15),Tensor:(1,256,15,15),Tensor:(1,256,15,15),Tensor:(1,256,8,8),Tensor:(1,256,4,4)}
        # ============  画特征图 =========================================================
        # show_feature_map(zf, name='z', index=0)

        self.zf = zf  #

    def track(self, x):
        xf = self.backbone(x)  #
        xf = self.fpn(xf)  # tuple:5 {Tensor:(1,256,31,31),Tensor:(1,256,31,31),Tensor:(1,256,31,31),Tensor:(1,256,16,16),Tensor:(1,256,8,8)}
        # =================画特征图 ======================
        # show_feature_map(xf, name='x', index=self.index)
        # self.index += 1
        # ===============================================

        cls, loc, filters = self.ban(self.zf, xf)  # cls: {Tensor:(1,2,17,17),Tensor:(1,2,9,9),Tensor:(1,2,5,5)}
        # loc:{Tensor:(1,4,17,17),Tensor:(1,4,9,9),Tensor:(1,4,5,5)}, filters: {Tensor:(1,1,17,17),Tensor:(1,1,9,9),Tensor:(1,1,5,5)}

        # cls1 = self.log_softmax(cls)
        # cls1 = cls1.detach().cpu().numpy()
        # cls_label1 = torch.ones_like(cls)
        # loc_label1 = torch.ones_like(loc)
        # cls1 = self.log_softmax(cls)
        # cls_loss1 = select_cross_entropy_loss(cls1, cls_label1)
        # loc_loss1 = select_iou_loss(loc, loc_label1, cls_label1)

        return {
                'cls': cls,
                'loc': loc,
                'filters': filters
               }


    def log_softmax(self, cls):
        if cfg.BAN.BAN:
            cls = cls.permute(0, 2, 3, 1).contiguous()
            cls = F.log_softmax(cls, dim=3)
        return cls

    def forward(self, data):  # data <c>
        """ only used in training
        """
        template = data['template'].cuda()  # <c> {Tensor:(28,3,127,127)}
        search = data['search'].cuda()  # <c> {Tensor:(28,3,255,255)}
        label_cls = data['label_cls']
        label_loc = data['label_loc']

        # get feature
        zf = self.backbone(template)  # <c> {list:3}  Tensor:(28,512,15,15),Tensor:(28,1024,15,15),Tensor:(28,2048,15,15),
        xf = self.backbone(search)  # <c> {list:3}  Tensor:(28,512,31,31),Tensor:(28,1024,31,31),Tensor:(28,2048,31,31),
        zf = self.fpn(zf)
        xf = self.fpn(xf)
        cls, loc, filters = self.ban(zf, xf)
        cls, loc, filters = permute_all_cls_and_box_to_N_HWA_K_and_concat(cls, loc, filters)
        label_cls, label_loc = permute_all_cls_and_box_to_N_HWA_K_and_concat(label_cls, label_loc, box_center=None, num_classes=1)
        label_cls = label_cls.cuda()
        label_cls = label_cls.to(torch.float32)
        label_loc = label_loc.cuda()
        cls = cls.sigmoid() * filters.sigmoid()
        cls = cls.softmax(1)[:, 0].unsqueeze(-1)
        cls_loss = focal_loss(cls, label_cls, alpha=cfg.FOCAL_LOSS.ALPHA,
                              gamma=cfg.FOCAL_LOSS.GAMMA, reduction=cfg.FOCAL_LOSS.REDUCTION)
        loc_loss = select_iou_loss(loc, label_loc, label_cls)



        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss  # 加权相加
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss

        return outputs
