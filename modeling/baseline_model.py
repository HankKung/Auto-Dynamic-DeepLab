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

class Cell_baseline(nn.Module):

    def __init__(self,
                BatchNorm,
                B, 
                prev_prev_C,
                prev_C, 
                cell_arch,
                network_arch,
                C_out,
                downup_sample):

        super(Cell_baseline, self).__init__()
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


class Model_1_baseline (nn.Module):
    def __init__(self,
                network_arch,
                cell_arch,
                num_classes,
                num_layers,
                BatchNorm,
                F=20,
                B=5,
                low_level_layer=1):

        super(Model_1_baseline, self).__init__()
        
        self.cells = nn.ModuleList()
        self.model_1_network = network_arch
        self.cell_arch = torch.from_numpy(cell_arch)
        self.num_model_1_layers = num_layers
        self._num_classes = num_classes
        self.low_level_layer = low_level_layer
        self.decoder_1 = Decoder(num_classes, BatchNorm)

        FB = F * B
        fm = {0: 1, 1: 2, 2: 4, 3: 8}

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            BatchNorm(64),
        )
        self.stem1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            BatchNorm(64),
        )

        self.stem2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            BatchNorm(128),
        )

        self.low_level_conv = nn.Sequential(nn.Conv2d(FB * 2**self.model_1_network[low_level_layer], 48, 1),
                                    BatchNorm(48),
                                    nn.ReLU())

        for i in range(self.num_model_1_layers):

            level = self.model_1_network[i]
            prev_level = self.model_1_network[i-1]
            prev_prev_level = self.model_1_network[i-2]

            downup_sample = int(prev_level - level)
            pre_downup_sample = int(prev_prev_level - level)
            if i == 0:
                downup_sample = int(0 - level)
                pre_downup_sample = int(-1 - level)
                _cell = Cell_baseline(BatchNorm,
                            B,
                            64,
                            128,                              
                            self.cell_arch,
                            self.model_1_network[i],
                            F * fm[level],                        
                            downup_sample) 
                
            elif i == 1:
                pre_downup_sample = int(0 - level)
                _cell = Cell_baseline(BatchNorm,
                            B,
                            128,
                            FB * fm[prev_level],
                            self.cell_arch,
                            self.model_1_network[i],
                            F * fm[level],
                            downup_sample)
            else:
                _cell = Cell_baseline(BatchNorm,
                            B, 
                            FB * fm[prev_prev_level],
                            FB * fm[prev_level],
                            self.cell_arch,
                            self.model_1_network[i],
                            F * fm[level],
                            downup_sample)
       
            self.cells += [_cell]

        if self.model_1_network[-1] == 1:
            mult = 2
        elif self.model_1_network[-1] == 2:
            mult =1

        self.aspp_1 = ASPP_train(FB * fm[self.model_1_network[-1]], \
                                      256, num_classes, BatchNorm, mult=mult)
        self._init_weight()

    def forward(self, x, target=None, criterion=criterion):
        size = (x.shape[2], x.shape[3])
        stem = self.stem0(x)
        stem0 = self.stem1(stem)
        stem1 = self.stem2(stem0)
        two_last_inputs = (stem0, stem1)

        for i in range(self.num_model_1_layers):       
            two_last_inputs = self.cells[i](
                two_last_inputs[0], two_last_inputs[1])

            if i == self.low_level_layer:
                low_level = two_last_inputs[1]
                low_level_feature = self.low_level_conv(low_level)

        x = two_last_inputs[-1]
        x = self.aspp_1(x)
        x = self.decoder_1(x, low_level_feature, size)

        if target == None:
            return low_level, two_last_inputs, x
        else:
            loss_1 = criterion(x, target)
            return low_level, two_last_inputs, loss_1 
            

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

class Model_2_baseline (nn.Module):
    def __init__(self, network_arch, cell_arch_1, cell_arch_2, num_classes, args, low_level_layer):
        super(Model_2_baseline, self).__init__()
        BatchNorm = SynchronizedBatchNorm2d if args.sync_bn == True else nn.BatchNorm2d
        F_2 = args.F_2
        F_1 = args.F_1
        B_2 = args.B_2
        B_1 = args.B_1
        num_model_1_layers = args.num_model_1_layers
        self.num_model_2_layers = len(network_arch) - num_model_1_layers
        self.cells = nn.ModuleList()
        self.model_2_network = network_arch[num_model_1_layers:]
        self.cell_arch_2 = torch.from_numpy(cell_arch_2)
        self._num_classes = num_classes

        model_1_network = network_arch[:args.num_model_1_layers]
        self.model_1 = Model_1_baseline(model_1_network, cell_arch_1, num_classes, num_model_1_layers, \
                                       BatchNorm, F=F_1, B=B_1, low_level_layer=low_level_layer)
        self.decoder_2 = Decoder(num_classes, BatchNorm)
        
   
        fm = {0: 1, 1: 2, 2: 4, 3: 8}
        for i in range(self.num_model_2_layers):

            level = self.model_2_network[i]
            prev_level = self.model_2_network[i-1]
            prev_prev_level = self.model_2_network[i-2]

            downup_sample = int(prev_level - level)

            if i == 0:
                downup_sample = int(model_1_network[-1] - self.model_2_network[0])
                pre_downup_sample = int(model_1_network[-2] - self.model_2_network[0])
                _cell = Cell_baseline(BatchNorm, B_2, 
                                        F_1 * B_1 * fm[model_1_network[-2]],
                                        F_1 * B_1 * fm[model_1_network[-1]],
                                        self.cell_arch_2,
                                        self.model_2_network[i],
                                        F_2 * fm[level],
                                        downup_sample)
            
            elif i == 1:
                pre_downup_sample = int(model_1_network[-1] - self.model_2_network[1])
                _cell = Cell_baseline(BatchNorm,
                                        B_2,
                                        F_1 * B_1 * fm[model_1_network[-1]],
                                        F_2 * B_2 * fm[self.model_2_network[0]],
                                        self.cell_arch_2,
                                        self.model_2_network[i],
                                        F_2 * fm[level],
                                        downup_sample)
            else:
                _cell = Cell_baseline(BatchNorm, B_2, 
                                        F_2 * B_2 * fm[prev_prev_level],
                                        F_2 * B_2 * fm[prev_level],
                                        self.cell_arch_2,
                                        self.model_2_network[i],
                                        F_2 * fm[level],
                                        downup_sample)

            self.cells += [_cell]

        if self.model_2_network[-1] == 1:
            mult = 2
        elif self.model_2_network[-1] == 2:
            mult =1

        self.low_level_conv = nn.Sequential(nn.Conv2d(F_1 * B_1 * 2**model_1_network[low_level_layer], 48, 1),
                                BatchNorm(48),
                                nn.ReLU())
        self.aspp_2 = ASPP_train(F_2 * B_2 * fm[self.model_2_network[-1]], 
                                     256, num_classes, BatchNorm, mult=mult)
        self._init_weight()

    def forward(self, x, target=None, criterion=None, evaluation=False):
        size = (x.shape[2], x.shape[3])
        if not evaluation:
            if target == None:
                low_level, two_last_inputs, y1 = self.model_1(x)

                for i in range(self.num_model_2_layers):
                    two_last_inputs = self.cells[i](
                        two_last_inputs[0], two_last_inputs[1])
                    
                y2 = two_last_inputs[-1]

                y2 = self.aspp_2(y2)
                low_level = self.low_level_conv(low_level)
                y2 = self.decoder_2(y2, low_level, size)     

                return y1, y2
            else:
                low_level, two_last_inputs, loss_1 = self.model_1(x, target=target, criterion=criterion)

                for i in range(self.num_model_2_layers):
                    two_last_inputs = self.cells[i](
                        two_last_inputs[0], two_last_inputs[1])
                    
                y2 = two_last_inputs[-1]

                y2 = self.aspp_2(y2)
                low_level = self.low_level_conv(low_level)
                y2 = self.decoder_2(y2, low_level, size)     
                loss_2 = criterion(y2, target)
                return loss_1, loss_2

        else:
            torch.cuda.synchronize()
            tic = time.perf_counter()

            low_level, two_last_inputs, y1 = self.device(x)

            torch.cuda.synchronize()
            tic_1 = time.perf_counter()

            for i in range(self.num_model_2_layers):
                two_last_inputs = self.cells[i](
                    two_last_inputs[0], two_last_inputs[1])

            y2 = two_last_inputs[-1]
            y2 = self.aspp_2(y2)
            low_level = self.low_level_conv(low_level)
            y2 = self.decoder_2(y2, low_level, size)

            torch.cuda.synchronize()
            tic_2 = time.perf_counter()

            return y1, y2, tic_1 - tic, tic_2 - tic

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
