
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modeling.genotypes import PRIMITIVES
from modeling.aspp_train import ASPP_train
from modeling.decoder import Decoder
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.operations import *

import time

class Cell_AutoDeepLab(nn.Module):

    def __init__(self,
                BatchNorm,
                B, 
                prev_prev_C,
                prev_C, 
                cell_arch,
                network_arch,
                C_out,
                downup_sample):

        super(Cell_AutoDeepLab, self).__init__()
        eps = 1e-5
        momentum = 0.1

        self.cell_arch = cell_arch
        self.downup_sample = downup_sample
        self.B = B

        self.pre_preprocess = ReLUConvBN(
            prev_prev_C, C_out, 1, 1, 0, BatchNorm, eps=eps, momentum=momentum, affine=True)
        self.preprocess = ReLUConvBN(
            prev_C, C_out, 1, 1, 0, BatchNorm, eps=eps, momentum=momentum, affine=True)

        self._ops = nn.ModuleList()
        if downup_sample == -1:
            self.preprocess = FactorizedReduce(prev_C, C_out, BatchNorm, eps=eps, momentum=momentum)
        elif downup_sample == 1:
            self.scale = 2
         
        for x in self.cell_arch:
            primitive = PRIMITIVES[x[1]]
            op = OPS[primitive](C_out, 1, BatchNorm, eps=eps, momentum=momentum, affine=True)
            self._ops.append(op)


    def scale_dimension(self, dim, scale):
        return int((float(dim) - 1.0) * scale + 1.0)


    def forward(self, prev_prev_input, prev_input):
        s1 = prev_input        
        if self.downup_sample == 1:
            feature_size_h = self.scale_dimension(
                s1.shape[2], self.scale)
            feature_size_w = self.scale_dimension(
                s1.shape[3], self.scale)
            s1 = F.interpolate(
                s1, [feature_size_h, feature_size_w], mode='bilinear')
        s1 = self.preprocess(s1)

        s0 = prev_prev_input
        del prev_prev_input

        s0 = F.interpolate(s0, [s1.shape[2], s1.shape[3]], mode='bilinear') \
            if s0.shape[2] != s1.shape[2] else s0
        s0 = self.pre_preprocess(s0)
        
        states = [s0, s1]
        offset = 0
        ops_index = 0
        for i in range(self.B):
            new_states = []
            for j, h in enumerate(states):
                branch_index = offset + j
                if branch_index in self.cell_arch[:, 0]:
                    new_state = self._ops[ops_index](h)
                    new_states.append(new_state)
                    ops_index += 1

            s = sum(new_states)
            offset += len(states)
            states.append(s)

        concat_feature = torch.cat(states[-self.B:], dim=1)
        return prev_input, concat_feature


class AutoDeepLab (nn.Module):
    def __init__(self, network_arch, cell_arch, num_classes, args, low_level_layer):
        super(AutoDeepLab, self).__init__()
        BatchNorm = SynchronizedBatchNorm2d if args.sync_bn == True else nn.BatchNorm2d
        F = args.F
        B = args.B
        self.num_model_layers = len(network_arch)
        self.cells = nn.ModuleList()
        self.model_network = network_arch
        self.cell_arch = torch.from_numpy(cell_arch)
        self.low_level_layer = low_level_layer
        self._num_classes = num_classes

        FB = F * B
        fm = {0: 1, 1: 2, 2: 4, 3: 8}
        self.stem0 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
            BatchNorm(64),
            nn.ReLU(inplace=True)

        )
        self.stem1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            BatchNorm(64),
        )

        self.stem2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            BatchNorm(128),
        )

        for i in range(self.num_model_layers):
            level = self.model_network[i]
            prev_level = self.model_network[i-1]
            prev_prev_level = self.model_network[i-2]

            downup_sample = int(prev_level - level)
            if i == 0:
                downup_sample = int(0 - level)
                pre_downup_sample = int(-1 - level)
                _cell = Cell_AutoDeepLab(BatchNorm,
                            B,
                            64,
                            128,                              
                            self.cell_arch,
                            self.model_network[i],
                            F * fm[level],                        
                            downup_sample) 
                
            elif i == 1:
                pre_downup_sample = int(0 - level)
                _cell = Cell_AutoDeepLab(BatchNorm,
                            B,
                            128,
                            FB * fm[prev_level],
                            self.cell_arch,
                            self.model_network[i],
                            F * fm[level],
                            downup_sample)
            else:
                _cell = Cell_AutoDeepLab(BatchNorm,
                            B, 
                            FB * fm[prev_prev_level],
                            FB * fm[prev_level],
                            self.cell_arch,
                            self.model_network[i],
                            F * fm[level],
                            downup_sample)

            self.cells += [_cell]

        if self.model_network[-1] == 1:
            mult = 2
        elif self.model_network[-1] == 2:
            mult =1

        self.low_level_conv = nn.Sequential(
                                nn.ReLU(),
                                nn.Conv2d(F * B * 2**self.model_network[low_level_layer], 48, 1, bias=False),
                                BatchNorm(48)
                                )
        
        self.aspp = ASPP_train(F * B * fm[self.model_network[-1]], 
                                256,
                                BatchNorm,
                                mult=mult)

        self.decoder = Decoder(num_classes, BatchNorm)
        self._init_weight()


    def forward(self, x, iter_rate=1.0):
        size = (x.shape[2], x.shape[3])
        stem = self.stem0(x)
        stem0 = self.stem1(stem)
        stem1 = self.stem2(stem0)
        two_last_inputs = (stem0, stem1)

        for i in range(self.num_model_layers):       
            two_last_inputs = self.cells[i](
                two_last_inputs[0], two_last_inputs[1])

            if i == self.low_level_layer:
                low_level = two_last_inputs[1]
                low_level = self.low_level_conv(low_level)
        y = two_last_inputs[-1]
        y = self.aspp(y)
        y = self.decoder(y, low_level, size)     

        return None, y

    def time_measure(self, x):
        size = (x.shape[2], x.shape[3])
        torch.cuda.synchronize()
        tic = time.perf_counter()
        stem = self.stem0(x)
        stem0 = self.stem1(stem)
        stem1 = self.stem2(stem0)
        two_last_inputs = (stem0, stem1)

        for i in range(self.num_model_layers):
            two_last_inputs = self.cells[i](
                two_last_inputs[0], two_last_inputs[1])
            if i == self.low_level_layer:
                low_level = two_last_inputs[1]
                low_level = self.low_level_conv(low_level)

        y = two_last_inputs[-1]
        y = self.aspp(y)
        y = self.decoder(y, low_level, size)

        torch.cuda.synchronize()
        tic_1 = time.perf_counter()

        return None, None, None, tic_1 - tic

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()