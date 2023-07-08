import os
import sys

# prj_path = os.path.join(os.path.dirname(__file__), '..')
# if prj_path not in sys.path:
#     sys.path.append(prj_path)

import argparse
import torch
from typing import Optional, List
from thop import profile
from thop.utils import clever_format
import time
import importlib
from torch import Tensor

from siamban.core.config import cfg
from siamban.models.model_builder import ModelBuilder
from siamban.tracker.tracker_builder import build_tracker
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


# -------------------------      手动赋值    --------------------------------------- #
args.config='D:/课题2/previous/code_use/siamban-master_new/experiments/siamban_r50_l234/config.yaml'
args.snapshot='D:/课题2/previous/code_use/siamban-master_new/experiments/siamban_r50_l234/model_my.pth'

def evaluate_vit(model, data):
    '''Speed Test'''
    macs1, params1 = profile(model, inputs=[data],
                             custom_ops=None, verbose=False)
    macs, params = clever_format([macs1, params1], "%.3f")
    print('overall macs is ', macs)
    print('overall params is ', params)

    T_w = 500
    T_t = 10
    print("testing speed ...")
    torch.cuda.synchronize()
    with torch.no_grad():
        # overall
        for i in range(T_w):
            _ = model(data)
        start = time.time()
        for i in range(T_t):
            _ = model(data)
        torch.cuda.synchronize()
        end = time.time()
        avg_lat = (end - start) / T_t
        print("The average overall latency is %.2f ms" % (avg_lat * 10))
        print("FPS is %.2f fps" % (1. / avg_lat))
        # for i in range(T_w):
        #     _ = model(template, search)
        # start = time.time()
        # for i in range(T_t):
        #     _ = model(template, search)
        # end = time.time()
        # avg_lat = (end - start) / T_t
        # print("The average backbone latency is %.2f ms" % (avg_lat * 1000))

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

if __name__ == '__main__':
    # load config
    ### ========= 注释掉 iou_loss 中的 44 ============
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()
    # load model
    # model = load_pretrain(model, args.snapshot).cuda().eval()
    # build tracker
    bs = 1
    z_sz = 127
    x_sz = 255
    response_map_size = [17, 9, 5]

    # get the template and search
    template = torch.randn(bs, 3, z_sz, z_sz)
    search = torch.randn(bs, 3, x_sz, x_sz)
    label_cls = [torch.randn(bs, 1, x, x) for x in response_map_size]
    label_loc = [torch.randn(bs, 4, x, x) for x in response_map_size]
    template = template.to(device)
    search = search.to(device)
    # label_cls = label_cls.to(device)
    # label_loc = label_loc.to(device)
    bbox = torch.randn(bs, 4)
    data = dict(template=template, search=search, label_cls=label_cls, label_loc=label_loc, bbox=bbox)
    model = model.to(device)
    template = template.to(device)
    search = search.to(device)
    evaluate_vit(model, data)



