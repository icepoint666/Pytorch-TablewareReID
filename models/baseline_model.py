# encoding: utf-8
"""
@author:  liaoxingyu
@contact: xyliao1993@qq.com 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import itertools

import torch.nn.functional as F
from torch import nn

from .resnet import ResNet



class ResNetBuilder(nn.Module):
    in_planes = 2048

    def __init__(self, last_stride=1, model_path='/DATA/model_zoo/resnet50-19c8e357.pth'):
        super().__init__()
        self.base = ResNet(last_stride)
        self.base.load_param(model_path)


    def forward(self, x):
        global_feat = self.base(x)
        global_feat = F.avg_pool2d(global_feat, global_feat.shape[2:])  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)

        return global_feat

    def get_optim_policy(self):
        base_param_group = self.base.parameters()

        return [
            {'params': base_param_group}
        ]
