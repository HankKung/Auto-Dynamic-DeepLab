import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.operations import ReLUConvBN

class ASPP_train(nn.Module):
    def __init__(self, C, out, BatchNorm, depth=256, conv=nn.Conv2d, eps=1e-5, momentum=0.1, mult=1):
        super(ASPP_train, self).__init__()
        self._C = C
        self._depth = depth
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.relu_non_inplace = nn.ReLU()
        self.aspp1 = conv(C, depth, kernel_size=1, stride=1, bias=False)
        self.aspp2 = conv(C, depth, kernel_size=3, stride=1,
                               dilation=int(6*mult), padding=int(6*mult), bias=False)
        self.aspp3 = conv(C, depth, kernel_size=3, stride=1,
                               dilation=int(12*mult), padding=int(12*mult), bias=False)
        self.aspp4 = conv(C, depth, kernel_size=3, stride=1,
                               dilation=int(18*mult), padding=int(18*mult), bias=False)
        self.aspp5 = conv(C, depth, kernel_size=1, stride=1, bias=False)

        self.conv1 = conv(depth * 5, out, kernel_size=1, stride=1, bias=False)
        
        self.bn1 = BatchNorm(out, eps=eps, momentum=momentum)
        self.aspp1_bn = BatchNorm(depth, eps=eps, momentum=momentum)
        self.aspp2_bn = BatchNorm(depth, eps=eps, momentum=momentum)
        self.aspp3_bn = BatchNorm(depth, eps=eps, momentum=momentum)
        self.aspp4_bn = BatchNorm(depth, eps=eps, momentum=momentum)
        self.aspp5_bn = BatchNorm(depth, eps=eps, momentum=momentum)

    def forward(self, x, confidence_map=None, iter_rate=1.0):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)

        x = self.relu_non_inplace(x)
        x1 = self.aspp1(x)
        x1 = self.aspp1_bn(x1)
        x1 = self.relu(x1)
        x2 = self.aspp2(x)
        x2 = self.aspp2_bn(x2)
        x2 = self.relu(x2)
        x3 = self.aspp3(x)
        x3 = self.aspp3_bn(x3)
        x3 = self.relu(x3)
        x4 = self.aspp4(x)
        x4 = self.aspp4_bn(x4)
        x4 = self.relu(x4)
        x5 = self.global_pooling(x)
        x5 = self.aspp5(x5)
        x5 = self.aspp5_bn(x5)
        x5 = self.relu(x5)
        x5 = nn.Upsample((x.shape[2], x.shape[3]), mode='bilinear',
                             align_corners=True)(x5)

        x = torch.cat((x1, x2, x3, x4, x5), 1)

        x = self.conv1(x)
        x = self.bn1(x)
        return x


class ASPP_Lite(nn.Module):
    def __init__(self, in_channels, low_level_channels, mid_channels, num_classes, BatchNorm):
        super().__init__()
        self._1x1_TL = ReLUConvBN(in_channels, mid_channels, 1, 1, 0, BatchNorm)
        self._1x1_BL = nn.Conv2d(in_channels, mid_channels, kernel_size=1)  # TODO: bias=False?
        self._1x1_TR = nn.Conv2d(mid_channels, num_classes, kernel_size=1)
        self._1x1_BR = nn.Conv2d(low_level_channels, num_classes, kernel_size=1)
        self.avgpool = torch.nn.AvgPool2d(kernel_size=49, stride=[16, 20], count_include_pad=False)

    def forward(self, x, low_level_feature):
        t1 = self._1x1_TL(x)
        B, C, H, W = t1.shape
        t2 = self.avgpool(x)
        t2 = self._1x1_BL(t2)
        t2 = torch.sigmoid(t2)
        t2 = F.interpolate(t2, size=(H, W), mode='bilinear', align_corners=False)
        t3 = t1 * t2
        h , w = int((float(t3.shape[2]) - 1.0) * 2 + 1.0), int((float(t3.shape[3]) - 1.0) * 2 + 1.0)
        t3 = F.interpolate(t3, [h, w], mode='bilinear', align_corners=False)
        t3 = self._1x1_TR(t3)
        t4 = self._1x1_BR(low_level_feature)
        return t3 + t4
