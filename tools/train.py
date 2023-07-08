# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import time
import math
import json
import random
import numpy as np
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.distributed import DistributedSampler

from siamban.utils.lr_scheduler import build_lr_scheduler
from siamban.utils.log_helper import init_log, print_speed, add_file_handler
from siamban.utils.distributed import dist_init, DistModule, reduce_gradients,\
        average_reduce, get_rank, get_world_size
from siamban.utils.model_load import load_pretrain, restore_from
from siamban.utils.average_meter import AverageMeter
from siamban.utils.misc import describe, commit
from siamban.models.model_builder import ModelBuilder
from siamban.datasets.dataset import BANDataset
from siamban.core.config import cfg


logger = logging.getLogger('global')
parser = argparse.ArgumentParser(description='siamese tracking')
parser.add_argument('--cfg', type=str, default='config.yaml',
                    help='configuration of tracking')
parser.add_argument('--seed', type=int, default=123456,
                    help='random seed')
parser.add_argument('--local_rank', type=int, default=0,
                    help='compulsory for pytorch launcer')
args = parser.parse_args()


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_data_loader():
    logger.info("build train dataset")
    # train_dataset
    if cfg.BAN.BAN:  # True
        train_dataset = BANDataset()  # {BANDataset:20000000}  <c>
    logger.info("build dataset done")

    train_sampler = None
    if get_world_size() > 1:  # True
        train_sampler = DistributedSampler(train_dataset)  # <c>
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.TRAIN.BATCH_SIZE,
                              num_workers=cfg.TRAIN.NUM_WORKERS,
                              pin_memory=True,
                              sampler=train_sampler)  # <c>
    return train_loader


def build_opt_lr(model, current_epoch=0):
    for param in model.backbone.parameters():  # param: <c> backbone里的模块，举例 conv1 bn1 依次从最底层拿出backbone的组成模块
        param.requires_grad = False
    for m in model.backbone.modules():  # m: <c> 举例 依次 Resnet() Conv2d BatchNorm2d 这个应该是从外到内
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    if current_epoch >= cfg.BACKBONE.TRAIN_EPOCH:  # False
        for layer in cfg.BACKBONE.TRAIN_LAYERS:
            for param in getattr(model.backbone, layer).parameters():
                param.requires_grad = True
            for m in getattr(model.backbone, layer).modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()

    trainable_params = []
    trainable_params += [{'params': filter(lambda x: x.requires_grad,
                                           model.backbone.parameters()),
                          'lr': cfg.BACKBONE.LAYERS_LR * cfg.TRAIN.BASE_LR}]  # <c> 可能是骨干网络和其他的部分 初始学习率不一样

    if cfg.ADJUST.ADJUST:  # True
        trainable_params += [{'params': model.fpn.parameters(),
                              'lr': cfg.TRAIN.BASE_LR}]  # <c>

    trainable_params += [{'params': model.ban.parameters(),
                          'lr': cfg.TRAIN.BASE_LR}]  # <c>

    optimizer = torch.optim.SGD(trainable_params,
                                momentum=cfg.TRAIN.MOMENTUM,
                                weight_decay=cfg.TRAIN.WEIGHT_DECAY)  # <c> 参考原论文中 实验第一段所述

    lr_scheduler = build_lr_scheduler(optimizer, epochs=cfg.TRAIN.EPOCH)  # <c>
    lr_scheduler.step(cfg.TRAIN.START_EPOCH)  # <c>
    return optimizer, lr_scheduler


def log_grads(model, tb_writer, tb_index):
    def weights_grads(model):
        grad = {}
        weights = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad[name] = param.grad
                weights[name] = param.data
        return grad, weights

    grad, weights = weights_grads(model)
    feature_norm, head_norm = 0, 0
    for k, g in grad.items():
        _norm = g.data.norm(2)
        weight = weights[k]
        w_norm = weight.norm(2)
        if 'feature' in k:
            feature_norm += _norm ** 2
        else:
            head_norm += _norm ** 2

        tb_writer.add_scalar('grad_all/'+k.replace('.', '/'),
                             _norm, tb_index)
        tb_writer.add_scalar('weight_all/'+k.replace('.', '/'),
                             w_norm, tb_index)
        tb_writer.add_scalar('w-g/'+k.replace('.', '/'),
                             w_norm/(1e-20 + _norm), tb_index)
    tot_norm = feature_norm + head_norm
    tot_norm = tot_norm ** 0.5
    feature_norm = feature_norm ** 0.5
    head_norm = head_norm ** 0.5

    tb_writer.add_scalar('grad/tot', tot_norm, tb_index)
    tb_writer.add_scalar('grad/feature', feature_norm, tb_index)
    tb_writer.add_scalar('grad/head', head_norm, tb_index)


def train(train_loader, model, optimizer, lr_scheduler, tb_writer):
    cur_lr = lr_scheduler.get_cur_lr()  # 0.001
    rank = get_rank()  # rank:0

    average_meter = AverageMeter()  # <csdn>

    def is_valid_number(x):
        return not(math.isnan(x) or math.isinf(x) or x > 1e4)  # 检查 数值 是否是 非数字nan 正无穷大 或者大于10000

    world_size = get_world_size()  # 全局的并行数  world_size:3
    num_per_epoch = len(train_loader.dataset) // \
        cfg.TRAIN.EPOCH // (cfg.TRAIN.BATCH_SIZE * world_size)  # num_per_epoch:11904 每个epoch执行的循环数
    start_epoch = cfg.TRAIN.START_EPOCH  # 0
    epoch = start_epoch  # 0

    if not os.path.exists(cfg.TRAIN.SNAPSHOT_DIR) and \
            get_rank() == 0:  # True
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR)

    logger.info("model\n{}".format(describe(model.module)))
    end = time.time()
    for idx, data in enumerate(train_loader):  # train_loader  data  <csdn>  以batch_size  28  拿出data
        if epoch != idx // num_per_epoch + start_epoch:  # False epoch更新时会是True
            epoch = idx // num_per_epoch + start_epoch  # 更新 epoch值

            if get_rank() == 0:  # 只在进程0上保存就行了，避免重复，而且保存的参数为 model.module
                torch.save(
                        {'epoch': epoch,
                         'state_dict': model.module.state_dict(),
                         'optimizer': optimizer.state_dict()},
                        cfg.TRAIN.SNAPSHOT_DIR+'/checkpoint_e%d.pth' % (epoch))

            if epoch == cfg.TRAIN.EPOCH:
                return

            if cfg.BACKBONE.TRAIN_EPOCH == epoch:  # 第10轮开始训练骨干网络
                logger.info('start training backbone.')
                optimizer, lr_scheduler = build_opt_lr(model.module, epoch)
                logger.info("model\n{}".format(describe(model.module)))

            lr_scheduler.step(epoch)
            cur_lr = lr_scheduler.get_cur_lr()
            logger.info('epoch: {}'.format(epoch+1))
        tb_idx = idx + start_epoch * num_per_epoch  # 0 tb_idx 表示当前epoch执行的轮数，与idx等价
        if idx % num_per_epoch == 0 and idx != 0:  # False 循环结束，标志当前epoch结束
            for idx, pg in enumerate(optimizer.param_groups):
                logger.info('epoch {} lr {}'.format(epoch+1, pg['lr']))
                if rank == 0:  # 写tensorboard
                    tb_writer.add_scalar('lr/group{}'.format(idx+1),
                                         pg['lr'], tb_idx)

        data_time = average_reduce(time.time() - end)
        if rank == 0:  # True
            tb_writer.add_scalar('time/data', data_time, tb_idx)

        outputs = model(data)  # <c>  数据送入模型 损失函数值
        loss = outputs['total_loss']

        if is_valid_number(loss.data.item()):  # True
            optimizer.zero_grad()
            loss.backward()
            reduce_gradients(model)  #

            if rank == 0 and cfg.TRAIN.LOG_GRADS:  # False
                log_grads(model.module, tb_writer, tb_idx)

            # clip gradient
            clip_grad_norm_(model.parameters(), cfg.TRAIN.GRAD_CLIP)
            optimizer.step()

        batch_time = time.time() - end
        batch_info = {}
        batch_info['batch_time'] = average_reduce(batch_time)
        batch_info['data_time'] = average_reduce(data_time)  # <c>
        for k, v in sorted(outputs.items()):  # 迭代拿出输出  k : 'cls_loss‘依次类推, v <c>
            batch_info[k] = average_reduce(v.data.item())  # batch_info <c>

        average_meter.update(**batch_info)  # <c>

        if rank == 0:  # True
            for k, v in batch_info.items():  # k:'batch_time','data_time','cls_loss'... v <c>
                tb_writer.add_scalar(k, v, tb_idx)

            if (idx+1) % cfg.TRAIN.PRINT_FREQ == 0:  # 这个满足条件时会触发 20打印一次
                info = "Epoch: [{}][{}/{}] lr: {:.6f}\n".format(
                            epoch+1, (idx+1) % num_per_epoch,
                            num_per_epoch, cur_lr)  # ’Epoch:[1][20/17857] lr:0.0010000\n‘
                for cc, (k, v) in enumerate(batch_info.items()):  # cc:索引， （k，v）与之前的一样
                    if cc % 2 == 0:
                        info += ("\t{:s}\t").format(
                                getattr(average_meter, k))  # ’Epoch:[1][20/17857] lr:0.0010000\n\tbatch_time:1.308527(1.368259)\t‘
                    else:
                        info += ("{:s}\n").format(
                                getattr(average_meter, k))  # ’Epoch:[1][20/17857] lr:0.0010000\n\tbatch_time:1.308527(1.368259)\tdata_time:0.488958(0.661270)\n‘
                logger.info(info)
                print_speed(idx+1+start_epoch*num_per_epoch,
                            average_meter.batch_time.avg,
                            cfg.TRAIN.EPOCH * num_per_epoch)
        end = time.time()


def main():
    rank, world_size = dist_init()  # world_size:3 rank: 1
    logger.info("init done")

    # load cfg
    cfg.merge_from_file(args.cfg)
    if rank == 0:  # True
        if not os.path.exists(cfg.TRAIN.LOG_DIR):
            os.makedirs(cfg.TRAIN.LOG_DIR)
        init_log('global', logging.INFO)
        if cfg.TRAIN.LOG_DIR:  # True
            add_file_handler('global',
                             os.path.join(cfg.TRAIN.LOG_DIR, 'logs.txt'),
                             logging.INFO)  # 同时打印日志到 文本文件和控制台

        logger.info("Version Information: \n{}\n".format(commit()))
        logger.info("config \n{}".format(json.dumps(cfg, indent=4)))

    # create model
    model = ModelBuilder().cuda().train()
    # dist_model = DistModule(model)

    # load pretrained backbone weights
    if cfg.BACKBONE.PRETRAINED:  # True
        cur_path = os.path.dirname(os.path.realpath(__file__))  # cur_path: '/root/data/zjx/siamBAN/siamban_ori/tools'
        backbone_path = os.path.join(cur_path, '../', cfg.BACKBONE.PRETRAINED)  # backbone_path:'/root/data/zjx/siamBAN/siamban_ori/tools/../pretrained_models/resnet50.model'
        load_pretrain(model.backbone, backbone_path)  # 到这里 model的骨干网络已经加载好预训练的模型了

    # create tensorboard writer
    if rank == 0 and cfg.TRAIN.LOG_DIR:  # True
        tb_writer = SummaryWriter(cfg.TRAIN.LOG_DIR)  # tensorboard
    else:
        tb_writer = None

    # build dataset loader
    train_loader = build_data_loader()

    # build optimizer and lr_scheduler
    optimizer, lr_scheduler = build_opt_lr(model,
                                           cfg.TRAIN.START_EPOCH)  # <c> 创建优化器，和学习率变化

    # resume training
    if cfg.TRAIN.RESUME:  # 断点续训  False
        logger.info("resume from {}".format(cfg.TRAIN.RESUME))
        assert os.path.isfile(cfg.TRAIN.RESUME), \
            '{} is not a valid file.'.format(cfg.TRAIN.RESUME)
        model, optimizer, cfg.TRAIN.START_EPOCH = \
            restore_from(model, optimizer, cfg.TRAIN.RESUME)
    # load pretrain
    elif cfg.TRAIN.PRETRAINED:  # Fasle
        load_pretrain(model, cfg.TRAIN.PRETRAINED)
    dist_model = DistModule(model)  # <c>  带有广播

    logger.info(lr_scheduler)
    logger.info("model prepare done")

    # start training
    train(train_loader, dist_model, optimizer, lr_scheduler, tb_writer)


if __name__ == '__main__':
    seed_torch(args.seed)
    main()
