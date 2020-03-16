import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class Decoder(nn.Module):

    def __init__(self, n_class, BatchNorm):
        super(Decoder, self).__init__()
        eps = 1e-5
        momentum = 0.1
        self._conv = nn.Sequential(
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                    BatchNorm(256, eps=eps, momentum=momentum),
                                    nn.ReLU(inplace=True),
                                    # 3x3 conv to refine the features
                                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                    BatchNorm(256, eps=eps, momentum=momentum),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, n_class, kernel_size=1, stride=1))

    def forward(self, x, low_level, size):
        x = F.interpolate(x, [low_level.shape[2], low_level.shape[3]], mode='bilinear') \
            if x.shape[2] != low_level.shape[2] else x
        x = torch.cat((x, low_level), 1)
        x = self._conv(x)
        x = F.interpolate(x, size, mode='bilinear')

        return x