# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging
import sys
import os

import cv2
import numpy as np
from torch.utils.data import Dataset

from siamban.utils.bbox import center2corner, Center
from siamban.datasets.point_target import PointTarget
from siamban.datasets.augmentation import Augmentation
from siamban.core.config import cfg

logger = logging.getLogger("global")

# setting opencv
pyv = sys.version[0]
if pyv[0] == '3':
    cv2.ocl.setUseOpenCL(False)


class SubDataset(object):
    def __init__(self, name, root, anno, frame_range, num_use, start_idx):
        cur_path = os.path.dirname(os.path.realpath(__file__))  # '/root/data/zjx/siamBAN/siamban_ori/siamban/datasets/'
        self.name = name  # 'COCO'
        self.root = os.path.join(cur_path, '../../', root)  # '/root/data/zjx/siamBAN/siamban_ori/siamban/datasets/../../training_dataset/coco/crop511'
        self.anno = os.path.join(cur_path, '../../', anno)  # '/root/data/zjx/siamBAN/siamban_ori/siamban/datasets/../../training_dataset/coco/train2017.json'
        self.frame_range = frame_range  # 1
        self.num_use = num_use  # 100000
        self.start_idx = start_idx  # 0
        logger.info("loading " + name)
        with open(self.anno, 'r') as f:  # <c>  打开注释的json文件
            meta_data = json.load(f)  # <c>  包含了所有目标注释的字典，每张 图片的路径名为键，值为图片中所有目标的bbox，这是经过预处理裁剪后的
            meta_data = self._filter_zero(meta_data)  # <c> 过滤掉没有目标的，以及 bbox长和宽为负值的图片

        for video in list(meta_data.keys()):  # video:'train2017/000000000009' 拿出COCO数据集一张图片裁剪后的，里面可能会包含多张，所以可以相当于一个视频
            for track in meta_data[video]:  # '00'  拿出裁剪后的图片中的一个目标 ，相当于视频中的某一帧了
                frames = meta_data[video][track]  # {'000000':[1.08,187.69,612.670000001,473.53]}
                frames = list(map(int,
                              filter(lambda x: x.isdigit(), frames.keys())))  # [0] 总是 0 好像，这个应该就只是一帧的图片，所以都是0
                frames.sort()
                meta_data[video][track]['frames'] = frames  # 在 字典里 添加 'frames':[0] ,这个也可以结合下面这个 if 条件判断是否是空的
                if len(frames) <= 0:  # False
                    logger.warning("{}/{} has no frames".format(video, track))
                    del meta_data[video][track]

        for video in list(meta_data.keys()):  # 'train2017/00000000009'  判断文件夹是否是空的
            if len(meta_data[video]) <= 0:  # False
                logger.warning("{} has no tracks".format(video))
                del meta_data[video]

        self.labels = meta_data  # <c>  标签
        self.num = len(self.labels)  # 117266
        self.num_use = self.num if self.num_use == -1 else self.num_use  # 100000
        self.videos = list(meta_data.keys())  # <c>  键拿出来，整成一列表，每张图片名称的路径
        logger.info("{} loaded".format(self.name))
        self.path_format = '{}.{}.{}.jpg'  # '{}.{}.{}.jpg'  这个后面会结合 format函数来对其进行赋值
        self.pick = self.shuffle()  # 打乱后只选 self.num_use 个

    def _filter_zero(self, meta_data):  # 过滤掉 w<0 h<0 的bbox
        meta_data_new = {}
        for video, tracks in meta_data.items():  # video={str}'train2017/000000000009' tracks : <c> //这里video相当于COCO中的每个图片，而tracks相当于每个图片中包含的所有目标的bbox
            new_tracks = {}
            for trk, frames in tracks.items():  # trk={str}'00'  frames={dict:1}{'000000':[1.08,187.69,612.6700000000001,473.53]}  // trk相当于目标的编号，frames相当于目标的bbox
                print("===================")
                print(frames)
                print("===================")
                new_frames = {}
                for frm, bbox in frames.items():  # frm:'000000'  bbox:[1.08,187.69,612.6700000000001,473.53]
                    if not isinstance(bbox, dict):  # True
                        if len(bbox) == 4:  # True
                            x1, y1, x2, y2 = bbox  # 看来bbox是 xyxy 型的
                            w, h = x2 - x1, y2 - y1  # 长 和 宽
                        else:
                            w, h = bbox
                        if w <= 0 or h <= 0:  # False
                            continue
                    new_frames[frm] = bbox
                if len(new_frames) > 0:  # True
                    new_tracks[trk] = new_frames
            if len(new_tracks) > 0:  # True
                meta_data_new[video] = new_tracks
        return meta_data_new

    def log(self):
        logger.info("{} start-index {} select [{}/{}] path_format {}".format(
            self.name, self.start_idx, self.num_use,
            self.num, self.path_format))

    def shuffle(self):
        lists = list(range(self.start_idx, self.start_idx + self.num))  #  {list:117266} 从0到117265，并且转成列表 <c>
        pick = []
        while len(pick) < self.num_use:  # 小于 使用的数量则循环 。若self.num_use小于lists 的长度则一次循环结束，截取这么长;若大于，则循环执行直至满足
            np.random.shuffle(lists)  # 随机打乱列表中的 索引顺序
            pick += lists
        return pick[:self.num_use]  # 只选 self.num_use 个 self.num_use即使超过长度，截取也只截取最大长度那么多。 而且这里保留的是索引，数值可能会超过100000

    def get_image_anno(self, video, track, frame):
        frame = "{:06d}".format(frame)  # '000000'
        image_path = os.path.join(self.root, video,
                                  self.path_format.format(frame, track, 'x'))  # '/root/data/zjx/siamBAN/siamban_ori/siamban/datasets/../../training_dataset/coco/crop511/train2017/000000525542/000000.01.x.jpg'
        image_anno = self.labels[video][track][frame]
        return image_path, image_anno  # 返回图片的路径 和 ground_truth 元组形式返回

    def get_positive_pair(self, index):
        video_name = self.videos[index]  # 'train2017/000000525542'  拿出对应索引的 图片的名称
        video = self.labels[video_name]  # vide0:{dict:5} <c>  拿出该图片所有目标对应的ground truth 信息
        track = np.random.choice(list(video.keys()))  # 01
        track_info = video[track]  # {'000000':[359.82,177.19,398.18,220.11],'frame':[0]}  <c> 拿出一个目标的信息

        frames = track_info['frames']  # [0]
        template_frame = np.random.randint(0, len(frames))  # 0
        left = max(template_frame - self.frame_range, 0)  # 0
        right = min(template_frame + self.frame_range, len(frames)-1) + 1  # 1
        search_range = frames[left:right]  #[0] 截取一段视频帧
        template_frame = frames[template_frame]  # 0  随机选取一帧
        search_frame = np.random.choice(search_range)  # 0  随机选取一帧
        return self.get_image_anno(video_name, track, template_frame), \
            self.get_image_anno(video_name, track, search_frame)

    def get_random_target(self, index=-1):
        if index == -1:
            index = np.random.randint(0, self.num)
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]
        frames = track_info['frames']
        frame = np.random.choice(frames)
        return self.get_image_anno(video_name, track, frame)

    def __len__(self):
        return self.num


class BANDataset(Dataset):
    def __init__(self,):
        super(BANDataset, self).__init__()

        # desired_size = (cfg.TRAIN.SEARCH_SIZE - cfg.TRAIN.EXEMPLAR_SIZE) / \
        #     cfg.POINT.STRIDE + 1 + cfg.TRAIN.BASE_SIZE
        # if desired_size != cfg.TRAIN.OUTPUT_SIZE:
        #     raise Exception('size not match!')

        # create point target
        self.point_target = PointTarget()

        # create sub dataset
        self.all_dataset = []
        start = 0
        self.num = 0
        for name in cfg.DATASET.NAMES:  # name: 'COCO' 这个就是拿出数据集的地方
            # a = cfg.DATASET  # 里面的内容由  config文件中设置的  详细见 <c>
            subdata_cfg = getattr(cfg.DATASET, name)  #  <c>
            sub_dataset = SubDataset(
                    name,
                    subdata_cfg.ROOT,
                    subdata_cfg.ANNO,
                    subdata_cfg.FRAME_RANGE,
                    subdata_cfg.NUM_USE,
                    start
                )
            start += sub_dataset.num  # 117266
            self.num += sub_dataset.num_use  # 100000

            sub_dataset.log()  # 类中方法，打印
            self.all_dataset.append(sub_dataset)  # <c>

        # data augmentation
        self.template_aug = Augmentation(
                cfg.DATASET.TEMPLATE.SHIFT,
                cfg.DATASET.TEMPLATE.SCALE,
                cfg.DATASET.TEMPLATE.BLUR,
                cfg.DATASET.TEMPLATE.FLIP,
                cfg.DATASET.TEMPLATE.COLOR
            )
        self.search_aug = Augmentation(
                cfg.DATASET.SEARCH.SHIFT,
                cfg.DATASET.SEARCH.SCALE,
                cfg.DATASET.SEARCH.BLUR,
                cfg.DATASET.SEARCH.FLIP,
                cfg.DATASET.SEARCH.COLOR
            )
        videos_per_epoch = cfg.DATASET.VIDEOS_PER_EPOCH  # 1000000  这个应该是每个epoch 训练的总batch数
        self.num = videos_per_epoch if videos_per_epoch > 0 else self.num  #  1000000
        self.num *= cfg.TRAIN.EPOCH  # 20000000  cfg.TRAIN.EPOCH 为 20，总的epoch
        self.pick = self.shuffle()  # <c> 里面放的是 随机打乱顺序的索引，总共20000000个，单个coco数据集循环了200次，算上epoch，等价于训练了200轮

    def shuffle(self):
        pick = []
        m = 0
        while m < self.num:  # 当m 小于时一直执行这个循环
            p = []
            for sub_dataset in self.all_dataset:
                sub_p = sub_dataset.pick  # {list：100000}
                p += sub_p  # 如果是单个数据集的话，p每次都是那些
            np.random.shuffle(p)
            pick += p
            m = len(pick)
        logger.info("shuffle done!")
        logger.info("dataset length {}".format(self.num))
        return pick[:self.num]

    def _find_dataset(self, index):
        for dataset in self.all_dataset:  # dataset <c>
            if dataset.start_idx + dataset.num > index:  # True 确定索引是否在范围之内 start_idx一般为0
                return dataset, index - dataset.start_idx

    def _get_bbox(self, image, shape):  # shape 就是 bbox
        imh, imw = image.shape[:2]  # 511,511
        if len(shape) == 4:  # True
            w, h = shape[2]-shape[0], shape[3]-shape[1]  # 举例 h:42.920000000000016  w:38.360000000000014
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = cfg.TRAIN.EXEMPLAR_SIZE  # 127
        wc_z = w + context_amount * (w+h)  # 举例 79.0000000000000
        hc_z = h + context_amount * (w+h)  # 举例 83.5600000000000
        s_z = np.sqrt(wc_z * hc_z)  # 举例 81.248015367668   裁剪填充的尺寸
        scale_z = exemplar_size / s_z  # 举例 1.5631115104248  ratio
        w = w*scale_z  # 举例 59.96150953989797  裁剪的尺寸 w 以及 h
        h = h*scale_z  # 举例 67.08890002743356
        cx, cy = imw//2, imh//2  # 举例 cx:255 cy : 255  以目标为中心
        bbox = center2corner(Center(cx, cy, w, h))  # {corner:4} <c>  原矩形框没有用上，只用了w和w计算一下比例，然后目标中心就是511图片的中心
        return bbox

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        a = index  # 举例 依次 6248367 ， 6094306 ， 4849304 || 第二次 依次 17136044, 11715241 , 8849483
        index = self.pick[index]  # 105839 但是第一个数总数这个，应该是设计了随机种子的关系
        dataset, index = self._find_dataset(index)  # index : 105839, dataset <c>

        gray = cfg.DATASET.GRAY and cfg.DATASET.GRAY > np.random.random()  # 0.0
        neg = cfg.DATASET.NEG and cfg.DATASET.NEG > np.random.random()  # False

        # get one dataset
        if neg:  # False
            template = dataset.get_random_target(index)
            search = np.random.choice(self.all_dataset).get_random_target()
        else:
            template, search = dataset.get_positive_pair(index)  # <c>   返回的是 模板和搜索图片对应的路径，在裁剪的文件夹下，以及ground_truth， 他们两个都一样的啊。它们可以一样，因为模板和搜索分支本来就来自于同一张图片，只不过模板图片小，搜索图片大
                                                                 # 都是一样的是因为 coco 数据集 的 所有的 frames 帧都为 [0],若是从视频中剪辑的话估计就不一样了
        # get image
        template_image = cv2.imread(template[0])  # ndarray:(511,511,3)
        search_image = cv2.imread(search[0])  # ndarray:(511,511,3)

        # get bounding box
        template_box = self._get_bbox(template_image, template[1])  # {Corner:4}  <c>  返回的是以模板输入尺寸确定的裁剪区域的大小
        search_box = self._get_bbox(search_image, search[1])  # {Corner:4}  <c>  它们俩都一样的

        # augmentation
        template, _ = self.template_aug(template_image,
                                        template_box,
                                        cfg.TRAIN.EXEMPLAR_SIZE,
                                        gray=gray)  # ndarry:(127,127,3)

        search, bbox = self.search_aug(search_image,
                                       search_box,
                                       cfg.TRAIN.SEARCH_SIZE,
                                       gray=gray)  # （255，255，3） ， {Corner:4} <c>

        # get labels
        cls, delta = self.point_target(bbox, cfg.POINT.TRAIN_OUTPUT_SIZE)
        template = template.transpose((2, 0, 1)).astype(np.float32)  # ndarray:(3,127,127)
        search = search.transpose((2, 0, 1)).astype(np.float32)  # ndarray:(3,255,255)
        return {
                'template': template,
                'search': search,
                'label_cls': cls,
                'label_loc': delta,
                'bbox': np.array(bbox)
                }
