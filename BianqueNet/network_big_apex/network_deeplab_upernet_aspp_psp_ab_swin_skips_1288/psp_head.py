import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
# from ..builder import HEADS
from .decode_head import BaseDecodeHead

import warnings

from mmcv.utils import Registry, build_from_cfg
from torch import nn

BACKBONES = Registry('backbone')
NECKS = Registry('neck')
HEADS = Registry('head')
LOSSES = Registry('loss')
SEGMENTORS = Registry('segmentor')


def build(cfg, registry, default_args=None):
    """Build a module.

    Args:
        cfg (dict, list[dict]): The config of modules, is is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        nn.Module: A built nn module.
    """

    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_backbone(cfg):
    """Build backbone."""
    return build(cfg, BACKBONES)


def build_neck(cfg):
    """Build neck."""
    return build(cfg, NECKS)


def build_head(cfg):
    """Build head."""
    return build(cfg, HEADS)


def build_loss(cfg):
    """Build loss."""
    return build(cfg, LOSSES)


def build_segmentor(cfg, train_cfg=None, test_cfg=None):
    """Build segmentor."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    return build(cfg, SEGMENTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))


class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg, align_corners):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvModule(
                        self.in_channels,
                        self.channels,
                        1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg)))

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = resize(
                ppm_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


@HEADS.register_module()
class PSPHead(BaseDecodeHead):
    """Pyramid Scene Parsing Network.

    This head is the implementation of
    `PSPNet <https://arxiv.org/abs/1612.01105>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(PSPHead, self).__init__(**kwargs)
        assert isinstance(pool_scales, (list, tuple))
        self.pool_scales = pool_scales
        self.psp_modules = PPM(
            self.pool_scales,
            self.in_channels,
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)
        output = self.cls_seg(output)
        return output
