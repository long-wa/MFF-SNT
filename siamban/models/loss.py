# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from siamban.core.config import cfg
from siamban.models.iou_loss import linear_iou


def get_cls_loss(pred, label, select):  # 通过这个来选出正负样本， label:(17500,) pred:(17500,2), select:(242,)
    if len(select.size()) == 0 or \
            select.size() == torch.Size([0]):
        return 0
    pred = torch.index_select(pred, 0, select)  # (242,2)
    label = torch.index_select(label, 0, select)  # (242,)
    return F.nll_loss(pred, label)


def select_cross_entropy_loss(pred, label):  # label:(28,25,25), pred:(28,25,25,2)
    pred = pred.view(-1, 2)  # (17500,2)
    label = label.view(-1)  # (17500,) 它的作用就是用来得到 下面的 pos 和 neg 的
    pos = label.data.eq(1).nonzero().squeeze().cuda()  # 正样本位置的索引
    neg = label.data.eq(0).nonzero().squeeze().cuda()  # 负样本位置的索引
    loss_pos = get_cls_loss(pred, label, pos)  #
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5


def weight_l1_loss(pred_loc, label_loc, loss_weight):
    if cfg.BAN.BAN:
        diff = (pred_loc - label_loc).abs()
        diff = diff.sum(dim=1)
    else:
        diff = None
    loss = diff * loss_weight
    return loss.sum().div(pred_loc.size()[0])


def select_iou_loss(pred_loc, label_loc, label_cls):
    label_cls = label_cls.reshape(-1)  # 17500
    pos = label_cls.data.eq(1).nonzero().squeeze().cuda()

    # pred_loc = pred_loc.permute(0, 2, 3, 1).reshape(-1, 4)  # (17500,4)
    pred_loc = torch.index_select(pred_loc, 0, pos)  # (242,4)

    # label_loc = label_loc.permute(0, 2, 3, 1).reshape(-1, 4)  # (17500,4)
    label_loc = torch.index_select(label_loc, 0, pos)  # (242,4)

    return linear_iou(pred_loc, label_loc)

def focal_loss(
    probs,
    targets,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",
):
    ce_loss = F.binary_cross_entropy(
        probs, targets, reduction="none"
    )
    p_t = probs * targets + (1 - probs) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss