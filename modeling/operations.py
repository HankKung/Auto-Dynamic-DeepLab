import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

OPS = {
  'none' : lambda C, stride, BatchNorm, eps, momentum, affine: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, BatchNorm, eps, momentum, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, stride, BatchNorm, eps, momentum, affine: nn.MaxPool2d(3, stride=stride, padding=1),
  'skip_connect' : lambda C, stride, BatchNorm, eps, momentum, affine: Identity(),
  'sep_conv_3x3' : lambda C, stride, BatchNorm, eps, momentum, affine: SepConv(C, C, 3, stride, 1, BatchNorm, eps=eps, momentum=momentum, affine=affine),
  'sep_conv_5x5' : lambda C, stride, BatchNorm, eps, momentum, affine: SepConv(C, C, 5, stride, 2, BatchNorm, eps=eps, momentum=momentum, affine=affine),
  'dil_conv_3x3' : lambda C, stride, BatchNorm, eps, momentum, affine: DilConv(C, C, 3, stride, 2, 2, BatchNorm, eps=eps, momentum=momentum, affine=affine),
  'dil_conv_5x5' : lambda C, stride, BatchNorm, eps, momentum, affine: DilConv(C, C, 5, stride, 4, 2, BatchNorm, eps=eps, momentum=momentum, affine=affine),
}

class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, BatchNorm, eps=1e-5, momentum=0.1, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      BatchNorm(C_out, eps=eps, momentum=momentum, affine=affine)
    )

  def forward(self, x):
    return self.op(x)


class DilConv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, BatchNorm, eps=1e-5, momentum=0.1, affine=True):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
      BatchNorm(C_out, eps=eps, momentum=momentum, affine=affine)
      )

  def forward(self, x):
    return self.op(x)


class SepConv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, BatchNorm, eps=1e-5, momentum=0.1, affine=True):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_out, C_out, kernel_size=1, padding=0, bias=False),
      BatchNorm(C_out, eps=eps, momentum=momentum, affine=affine),
      nn.ReLU(inplace=False),
      nn.Conv2d(C_out, C_out, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_out, C_out, kernel_size=1, padding=0, bias=False),
      BatchNorm(C_out, eps=eps, momentum=momentum, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)


class FactorizedReduce(nn.Module):
  def __init__(self, C_in, C_out, BatchNorm, eps=1e-5, momentum=0.1, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.bn = BatchNorm(C_out, eps=eps, momentum=momentum, affine=affine)
    self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)

  def forward(self, x):
    x = self.relu(x)
    y = self.pad(x)
    out = torch.cat([self.conv_1(x), self.conv_2(y[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out


class DoubleFactorizedReduce(nn.Module):
  def __init__(self, C_in, C_out, BatchNorm, eps=1e-5, momentum=0.1, affine=True):
    super(DoubleFactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=4, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=4, padding=0, bias=False)
    self.bn = BatchNorm(C_out, affine=affine)
    self.pad = nn.ConstantPad2d((0, 2, 0, 2), 0)

  def forward(self, x):
    x = self.relu(x)
    y = self.pad(x)
    out = torch.cat([self.conv_1(x), self.conv_2(y[:, :, 2:, 2:])], dim=1)
    out = self.bn(out)
    return out


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, paddings, dilations, BatchNorm=nn.BatchNorm2d, momentum=0.0003):

        super(ASPP, self).__init__()
        self.relu = nn.ReLU()
        self.conv11 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False),
                                    BatchNorm(in_channels),
                                    nn.ReLU(inplace=True))
        self.conv33 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3,
                                    padding=paddings, dilation=dilations, bias=False),
                                    BatchNorm(in_channels),
                                    nn.ReLU(inplace=True))
        self.conv_p = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False),
                                    BatchNorm(in_channels),
                                    nn.ReLU(inplace=True))

        self.concate_conv = nn.Sequential(nn.Conv2d(in_channels * 3, in_channels, 1, bias=False,  stride=1, padding=0),
                                          BatchNorm(in_channels),
                                          nn.ReLU(inplace=True))
        self.final_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False,  stride=1, padding=0)
        
    def forward(self, x):
        x = self.relu(x)
        conv11 = self.conv11(x)
        conv33 = self.conv33(x)

        # image pool and upsample
        image_pool = nn.AdaptiveAvgPool2d(1)
        upsample = nn.Upsample(size=x.size()[2:], mode='bilinear', align_corners=True)
        image_pool = image_pool(x)
        conv_image_pool = self.conv_p(image_pool)
        upsample = upsample(conv_image_pool)

        # concate
        concate = torch.cat([conv11, conv33, upsample], dim=1)
        concate = self.concate_conv(concate)
        return self.final_conv(concate)


def normalized_shannon_entropy(x, num_class=19):
    size = (x.shape[2], x.shape[3])
    x = F.softmax(x, dim=1).permute(0, 2, 3, 1) * F.log_softmax(x, dim=1).permute(0, 2, 3, 1)
    x = torch.sum(x, dim=3)
    x = x / math.log(num_class)
    x = -x

    x = x.sum()
    x = x / (size[0] * size[1])
    return x.item()

def confidence_max(x, thresold, num_class=19):
    x = F.softmax(x, dim=1)
    size = (x.shape[2], x.shape[3])
    max_map = torch.max(x, 1)
    max_map = max_map[0]
    max_map = max_map[max_map > thresold]
    num_max = max_map.shape[0]
    num_max = num_max / (size[0] * size[1])
    return num_max
  