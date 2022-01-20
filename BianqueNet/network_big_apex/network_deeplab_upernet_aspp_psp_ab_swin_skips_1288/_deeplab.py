import torch
from torch import nn
from torch.nn import functional as F

from .utils import _SimpleSegmentationModel
from .swin_transform_skips import swin_skips, swin_feature
import cv2

__all__ = ["DeepLabV3"]


class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass

class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        # self.project = nn.Sequential(
        #     nn.Conv2d(low_level_channels, 48, 1, bias=False),
        #     nn.BatchNorm2d(48),
        #     nn.ReLU(inplace=True),
        # )
        # # swin特征提取
        # self.swin_8d_feature = swin_feature(in_channels=64, hidden_dim=8, downscaling_factors=8, heads=3)
        # self.swin_4d_feature = swin_feature(in_channels=256, hidden_dim=32, downscaling_factors=4, heads=3)
        # self.swin_2d_feature = swin_feature(in_channels=512, hidden_dim=64, downscaling_factors=2, heads=3)
        # # 深度特征提取
        self.aspp = ASPP(in_channels*2, aspp_dilate)
        self.psp = PSP(in_channels, pool_sizes=[1, 2, 3, 6])
        # 跳跃连接处理
        self.skip_conv1 = SkipConv(64, 4)
        self.skip_layer1 = SkipConv(256, 16)
        self.skip_layer2 = SkipConv(512, 32)
        # swin的跳跃连接处理
        self.swin_skip_conv1 = swin_skips(in_channels=64, hidden_dim=4)
        self.swin_skip_layer1 = swin_skips(in_channels=256, hidden_dim=16)
        self.swin_skip_layer2 = swin_skips(in_channels=512, hidden_dim=32)
        # 上采样  64to128和128to256两个上采样使用转置卷积来调节通道数目
        # self.up64to128 = nn.ConvTranspose2d(320, 256, 2, stride=2)
        # self.up128to256 = nn.ConvTranspose2d(288, 256, 2, stride=2)

        # FPN层
        self.in_fpn_channels = [360, 352, 320, 256]
        # self.fpn_32 = Fpn_Conv(360, 128)
        self.fpn_64 = Fpn_Conv(320, 320)
        self.fpn_128 = Fpn_Conv(352, 352)
        self.fpn_256 = Fpn_Conv(360, 360)
        # # cat之后再做一个3*3卷积
        # self.fpn_bottleneck = Fpn_Conv(sum(self.in_fpn_channels), 256)

        self.classifier = nn.Sequential(
            nn.Conv2d(sum(self.in_fpn_channels), 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        # print('layer_2: ', feature['layer_2'].shape)
        # print('layer_3: ', feature['layer_3'].shape)
        # print('low_level: ', feature['low_level'].shape)
        # # swin 特征提取
        # swin_8d_feature = self.swin_8d_feature(feature['conv1'])
        # swin_4d_feature = self.swin_4d_feature(feature['low_level'])
        # swin_2d_feature = self.swin_2d_feature(feature['layer_2'])
        # 卷积的跳跃连接
        # conv1_feature = self.skip_conv1(feature['conv1'])
        # layer1_feature = self.skip_layer1(feature['low_level'])
        skip_feature = self.skip_layer2(feature['layer_2'])
        
        # 打印特征图
        # print('out', feature['out'].shape)
        # pic_print = feature['out']
        # pic_print = torch.squeeze(pic_print).numpy()
        # pic_print = pic_print.transpose([1, 2, 0])  # w*h*c
        # pic_print = pic_print[:, :, 0:3]
        # pic_print = 255*(pic_print-pic_print.min()) / (pic_print.max()-pic_print.min())
        # cv2.imwrite('F:\\yaozhui_tesla\\bianque_14\\ct_data\\shiyan\\mid_conv_feature\\backbone_feature.png', pic_print)
        # cv2.imshow('backbone_feature:', pic_print)

        # swin的跳跃连接
        # swin_conv1_feature = self.swin_skip_conv1(feature['conv1'])
        # swin_layer1_feature = self.swin_skip_layer1(feature['low_level'])
        swin_skip_feature = self.swin_skip_layer2(feature['layer_2'])
        # low_level_feature = self.project(feature['low_level'])
        # print('out: ', feature['out'].shape)

        # 深度特征提取
        output_feature = self.psp(feature['out'])
        # print('output_feature: ', output_feature.shape)
        output_feature = self.aspp(output_feature)
        # 打印特征图
        # pic_print_1 = output_feature
        # pic_print_1 = torch.squeeze(pic_print_1).numpy()
        # pic_print_1 = pic_print_1.transpose([1, 2, 0])  # w*h*c
        # pic_print_1 = pic_print_1[:, :, 0:3]
        # pic_print_1 = 255*(pic_print_1-pic_print_1.min()) / (pic_print_1.max()-pic_print_1.min())
        # cv2.imwrite('F:\\yaozhui_tesla\\bianque_14\\ct_data\\shiyan\\mid_conv_feature\\deep_feature.png', pic_print_1)
        # cv2.imshow('deep_feature:', pic_print_1)

        # cat深度特征和swin特征
        # output_feature = torch.cat([output_feature, swin_8d_feature, swin_4d_feature, swin_2d_feature], dim=1)
        # build top-down path
        # up32to64
        output_feature_64 = F.interpolate(output_feature, size=skip_feature.shape[2:], mode='bilinear', align_corners=False)
        # print('output_feature_64: ', output_feature_64.shape)
        output_feature_64 = torch.cat([skip_feature,
                                       swin_skip_feature,
                                       output_feature_64], dim=1)
        # del layer2_feature
        # print('output_feature_64: ', output_feature_64.shape)
        skip_feature = self.skip_layer1(feature['low_level'])
        swin_skip_feature = self.swin_skip_layer1(feature['low_level'])
        # up64to128
        # output_feature_128 = self.up64to128(output_feature_64)
        output_feature_128 = F.interpolate(output_feature_64, size=skip_feature.shape[2:], mode='bilinear', align_corners=False)
        # print('output_feature_128: ', output_feature_128.shape)
        output_feature_128 = torch.cat([skip_feature,
                                        swin_skip_feature,
                                        output_feature_128], dim=1)
        # del layer1_feature
        # print('output_feature_128: ', output_feature_128.shape)
        skip_feature = self.skip_conv1(feature['conv1'])
        swin_skip_feature = self.swin_skip_conv1(feature['conv1'])
        # up128to256
        # output_feature_256 = self.up128to256(output_feature_128)
        output_feature_256 = F.interpolate(output_feature_128, size=skip_feature.shape[2:], mode='bilinear', align_corners=False)
        output_feature_256 = torch.cat([skip_feature,
                                        swin_skip_feature,
                                        output_feature_256], dim=1)
        # del conv1_feature
        del skip_feature
        del swin_skip_feature
        del feature
        # print('output_feature_256: ', output_feature_256.shape)
        # print('output:', output_feature.shape)

        # build outputs FPN
        # output_fpn_32 = self.fpn_32(output_feature)
        output_feature_64 = self.fpn_64(output_feature_64)
        output_feature_128 = self.fpn_128(output_feature_128)
        output_feature_256 = self.fpn_256(output_feature_256)
        # concat FPN
        output_feature = F.interpolate(output_feature, size=output_feature_256.shape[2:], mode='bilinear',
                                      align_corners=False)
        output_feature_64 = F.interpolate(output_feature_64, size=output_feature_256.shape[2:], mode='bilinear',
                                      align_corners=False)
        output_feature_128 = F.interpolate(output_feature_128, size=output_feature_256.shape[2:], mode='bilinear',
                                      align_corners=False)
        # output_cat_fpn = torch.cat([output_feature, output_feature_64, output_feature_128,
        #                             output_feature_256], dim=1)
        # output_cat_fpn = output_fpn_32+output_fpn_64+output_fpn_128+output_fpn_256
        return self.classifier(torch.cat([output_feature, output_feature_64, output_feature_128,
                                    output_feature_256], dim=1))
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHead, self).__init__()

        self.classifier = nn.Sequential(
            ASPP(in_channels, aspp_dilate),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        return self.classifier( feature['out'] )

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                            stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d( in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=in_channels ),
            # PointWise Conv
            nn.Conv2d( in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )
        
        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class PSP(nn.Module):
    def __init__(self, in_channels, pool_sizes):
        super(PSP, self).__init__()
        out_channels = in_channels // len(pool_sizes)
        # -----------------------------------------------------#
        #   分区域进行平均池化
        #   30, 30, 320 + 30, 30, 80 + 30, 30, 80 + 30, 30, 80 + 30, 30, 80 = 30, 30, 640
        # -----------------------------------------------------#
        self.stages = nn.ModuleList(
            [self._make_stages(in_channels, out_channels, pool_size) for pool_size in pool_sizes])

    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        # print('features', features.shape)
        pyramids = [features]
        # 这个F.interpolate函数应该是resize的作用
        pyramids.extend(
            [F.interpolate(stage(features), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages])
        output = torch.cat(pyramids, dim=1)
        # print('output', output.shape)
        return output


class SkipConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SkipConv, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.project(input)


class Fpn_Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Fpn_Conv, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3,padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.project(input)

def convert_to_separable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0]>1:
        new_module = AtrousSeparableConvolution(module.in_channels,
                                      module.out_channels, 
                                      module.kernel_size,
                                      module.stride,
                                      module.padding,
                                      module.dilation,
                                      module.bias)
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module