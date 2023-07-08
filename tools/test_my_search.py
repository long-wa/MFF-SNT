# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

from got10k.experiments import *

import cv2
import torch
import numpy as np
import sys

from siamban.core.config import cfg
from siamban.models.model_builder import ModelBuilder
from siamban.tracker.tracker_1 import build_tracker
from siamban.utils.model_load import load_pretrain

parser = argparse.ArgumentParser(description='siamese tracking')
parser.add_argument('--dataset', type=str,
        help='datasets')
parser.add_argument('--config', default='', type=str,
        help='config file')
parser.add_argument('--snapshot', default='', type=str,
        help='snapshot of models to eval')
parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--vis', action='store_true',
        help='whether visualzie result')
parser.add_argument('--gpu_id', default='not_set', type=str,
        help="gpu id")

args = parser.parse_args()

if args.gpu_id != 'not_set':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

torch.set_num_threads(1)

def main():
    # load config
    cfg.merge_from_file(args.config)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(cur_dir, '../testing_dataset', args.dataset)

    # create model
    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build tracker
    tracker = build_tracker(model)

    e = ExperimentOTB(dataset_root, version=2015)
    e.run(tracker)
    e.report([tracker.name])

if __name__=='__main__':
    main()

