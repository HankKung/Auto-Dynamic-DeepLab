import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from modeling.genotypes import PRIMITIVES
from modeling.aspp_train import *
from modeling.decoder import Decoder
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.operations import *

import time

class Cell(nn.Module):

    def __init__(self,
                BatchNorm,
                B, 
                prev_prev_C,
                prev_C, 
                cell_arch,
                network_arch,
                C_out,
                downup_sample,
                dense_in=False,
                dense_out=True):

        super(Cell, self).__init__()
        eps = 1e-5
        momentum = 0.1

        self.cell_arch = cell_arch
        self.downup_sample = downup_sample
        self.B = B
        self.dense_in = dense_in
        self.dense_out = dense_out

        self.preprocess = ReLUConvBN(
            prev_C, C_out, 1, 1, 0, BatchNorm, eps=eps, momentum=momentum, affine=True)

        self._ops = nn.ModuleList()
        if downup_sample == -1:
            self.preprocess = FactorizedReduce(prev_C, C_out, BatchNorm, eps=eps, momentum=momentum)
        elif downup_sample == 1:
            self.scale = 2
         
        if self.dense_in == True:
            self.pre_preprocess = nn.ModuleList()
            for i in range(len(prev_prev_C)):
                self.pre_preprocess.append(
                    ReLUConvBN(prev_prev_C[i], C_out, 1, 1, 0, BatchNorm, eps=eps, momentum=momentum, affine=True))
            self.pre_preprocess_1x1 = ReLUConvBN(len(prev_prev_C) * C_out, C_out, 1, 1, 0, BatchNorm, eps=eps, momentum=momentum, affine=True)
        else:
            self.pre_preprocess = ReLUConvBN(
                prev_prev_C, C_out, 1, 1, 0, BatchNorm, eps=eps, momentum=momentum, affine=True)
        if self.dense_out == True:
            self.dense_process = ReLUConvBN(C_out * B, C_out, 1, 1, 0, BatchNorm, eps=eps, momentum=momentum, affine=True)

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

        if self.dense_in != True:
            s0 = F.interpolate(s0, [s1.shape[2], s1.shape[3]], mode='bilinear') \
                    if s0.shape[2] != s1.shape[2] else s0
            s0 = self.pre_preprocess(s0)
        else:
            for i in range(len(s0)):
                s0[i] = F.interpolate(s0[i], [s1.shape[2], s1.shape[3]], mode='bilinear') \
                    if s0[i].shape[2] != s1.shape[2] else s0[i]
                s0[i] = self.pre_preprocess[i](s0[i])
            s0 = torch.cat(s0, dim=1)
            s0 = self.pre_preprocess_1x1(s0)
        states = [s0, s1]
        del s0, s1

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
        if self.dense_out:
            return prev_input, concat_feature, self.dense_process(concat_feature)
        else:
            return concat_feature

class Model_1 (nn.Module):
    def __init__(self,
                network_arch,
                cell_arch,
                num_classes,
                num_layers,
                BatchNorm,
                F=20,
                B=5,
                low_level_layer=0):

        super(Model_1, self).__init__()
        
        self.cells = nn.ModuleList()
        self.model_1_network = network_arch
        self.cell_arch = torch.from_numpy(cell_arch)
        self.num_model_1_layers = num_layers
        self.low_level_layer = low_level_layer
        self._num_classes = num_classes

        FB = F * B
        fm = {0: 1, 1: 2, 2: 4, 3: 8}

        eps = 1e-5
        momentum = 0.1

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
            BatchNorm(64, eps=eps, momentum=momentum),
            nn.ReLU(inplace=True)
        )

        self.stem1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            BatchNorm(64, eps=eps, momentum=momentum),
        )

        self.stem2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            BatchNorm(128, eps=eps, momentum=momentum)
        )


        for i in range(self.num_model_1_layers):
            level = self.model_1_network[i]
            prev_level = self.model_1_network[i-1]
            prev_prev_level = self.model_1_network[i-2]

            downup_sample = int(prev_level - level)

            if i == 0:
                downup_sample = int(0 - level)
                _cell = Cell(BatchNorm,
                            B,
                            64,
                            128,                               
                            self.cell_arch,
                            self.model_1_network[i],
                            F * fm[level],                        
                            downup_sample) 
                
            elif i == 1:
                _cell = Cell(BatchNorm,
                            B,
                            128,
                            FB * fm[prev_level],
                            self.cell_arch,
                            self.model_1_network[i],
                            F * fm[level],
                            downup_sample)
            elif i == 2:
                _cell = Cell(BatchNorm,
                            B, 
                            FB * fm[prev_prev_level],
                            FB * fm[prev_level],
                            self.cell_arch,
                            self.model_1_network[i],
                            F * fm[level],
                            downup_sample)
            else:
                dense_channel_list = [F * fm[stride] for stride in self.model_1_network[:i-1]]
                _cell = Cell(BatchNorm,
                            B, 
                            dense_channel_list,
                            FB * fm[prev_level],
                            self.cell_arch,
                            self.model_1_network[i],
                            F * fm[level],
                            downup_sample,
                            dense_in=True)
       
            self.cells += [_cell]

        if self.model_1_network[-1] == 1:
            mult = 2
        elif self.model_1_network[-1] == 2:
            mult =1

        self._init_weight()


    def forward(self, x):
        size = (x.shape[2], x.shape[3])
        stem = self.stem0(x)
        stem0 = self.stem1(stem)
        stem1 = self.stem2(stem0)
        two_last_inputs = [stem0, stem1]
        dense_feature_map = []

        for i in range(self.num_model_1_layers):       
            if i < 3:
                two_last_inputs[0], two_last_inputs[1], feature_map = self.cells[i](
                    two_last_inputs[0], two_last_inputs[1])
                dense_feature_map.append(feature_map)

            else:
                _, x, feature_map = self.cells[i](dense_feature_map[:-1], x)
                dense_feature_map.append(feature_map)

            if i == self.low_level_layer:
                low_level_feature = two_last_inputs[1]

            if i == 2:
                x = two_last_inputs[1]


        return low_level_feature, dense_feature_map, x


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


class Model_2 (nn.Module):
    def __init__(self,
                network_arch,
                cell_arch, 
                num_classes, 
                args, 
                low_level_layer):

        super(Model_2, self).__init__()
        BatchNorm = SynchronizedBatchNorm2d if args.sync_bn == True else nn.BatchNorm2d
        F = args.F
        B = args.B

        eps = 1e-5
        momentum = 0.1

        self.args = args
        num_model_1_layers = args.num_model_1_layers
        self.num_model_2_layers = len(network_arch) - num_model_1_layers

        self.cells = nn.ModuleList()
        self.model_2_network = network_arch[num_model_1_layers:]
        self.cell_arch = torch.from_numpy(cell_arch)
        self._num_classes = num_classes

        model_1_network = network_arch[:args.num_model_1_layers]
        self.model_1 = Model_1(model_1_network, cell_arch, num_classes, num_model_1_layers, \
                                       BatchNorm, F=F, B=B, low_level_layer=low_level_layer)
        self.decoder = Decoder(num_classes, BatchNorm)
          
        fm = {0: 1, 1: 2, 2: 4, 3: 8}
        for i in range(self.num_model_2_layers):

            level = self.model_2_network[i]
            prev_level = self.model_2_network[i-1]

            downup_sample = int(prev_level - level)
            dense_channel_list_1 = [F * fm[stride] for stride in model_1_network]

            if i == 0:
                downup_sample = int(model_1_network[-1] - self.model_2_network[0])
                dense_channel_list_2 = dense_channel_list_1[:-1]
                _cell = Cell(BatchNorm,
                            B, 
                            dense_channel_list_2,
                            F * B * fm[model_1_network[-1]],
                            self.cell_arch,
                            self.model_2_network[i],
                            F * fm[level],
                            downup_sample,
                            dense_in=True)
            
            elif i == 1:
                dense_channel_list_2 = dense_channel_list_1
                _cell = Cell(BatchNorm, B,
                            dense_channel_list_2,
                            F * B * fm[self.model_2_network[0]],
                            self.cell_arch,
                            self.model_2_network[i],
                            F * fm[level],
                            downup_sample,
                            dense_in=True)

            elif i < self.num_model_2_layers - 2:
                dense_channel_list_2 = dense_channel_list_1 + \
                                        [F * fm[stride] for stride in self.model_2_network[:i-1]]
                _cell = Cell(BatchNorm,
                            B, 
                            dense_channel_list_2,
                            F * B * fm[prev_level],
                            self.cell_arch,
                            self.model_2_network[i],
                            F * fm[level],
                            downup_sample,
                            dense_in=True)

            else:
                dense_channel_list_2 = dense_channel_list_1 + \
                                        [F * fm[stride] for stride in self.model_2_network[:i-1]]
                _cell = Cell(BatchNorm,
                            B, 
                            dense_channel_list_2,
                            F * B * fm[prev_level],
                            self.cell_arch,
                            self.model_2_network[i],
                            F * fm[level],
                            downup_sample,
                            dense_in=True,
                            dense_out=False)

            self.cells += [_cell]

        if self.model_2_network[-1] == 1:
            mult = 2
        elif self.model_2_network[-1] == 2:
            mult =1

        self.low_level_conv = nn.Sequential(
                                    nn.ReLU(),
                                    nn.Conv2d(F * B * 2**model_1_network[low_level_layer], 48, 1, bias=False),
                                    BatchNorm(48, eps=eps, momentum=momentum),
                                    )
        self.aspp = ASPP_train(F * B * fm[self.model_2_network[-1]], 
                                512,
                                BatchNorm,
                                mult=mult,
                                use_oc=self.args.use_oc)
        self._init_weight()


    def forward(self, x, evaluation=False, threshold=None):
        size = (x.shape[2], x.shape[3])
        if not evaluation:
            low_level, dense_feature_map, x = self.model_1(x)
            low_level = self.low_level_conv(low_level)

            y1 = self.aspp(x)
            y1 = self.decoder(y1, low_level, size)

            if self.args.use_oc and self.args.confidence_map:
                confidence_map = normalized_shannon_entropy(y1, get_value=False)

            for i in range(self.num_model_2_layers):
                if i < self.num_model_2_layers - 2:
                    _, x, feature_map = self.cells[i](dense_feature_map[:-1], x)
                    dense_feature_map.append(feature_map)
                elif i == self.num_model_2_layers -1:
                    x = self.cells[i](dense_feature_map, x)
                else:
                    x = self.cells[i](dense_feature_map[:-1], x)

            del feature_map, dense_feature_map

            if self.args.confidence_map:
                x = self.aspp(x, confidence_map)
            else:
                x = self.aspp(x)
            x = self.decoder(x, low_level, size)     

            return y1, x

        elif evaluation and threshold == None:
            torch.cuda.synchronize()
            tic = time.perf_counter()

            low_level, dense_feature_map, x = self.model_1(x)
            low_level = self.low_level_conv(low_level)
            y1 = self.aspp(x)
            y1 = self.decoder(y1, low_level, size)     

            if self.args.use_oc and self.args.confidence_map:
                confidence_map, confidence = normalized_shannon_entropy(y1, get_value=True)

            torch.cuda.synchronize()
            tic_1 = time.perf_counter()

            for i in range(self.num_model_2_layers):
                if i < self.num_model_2_layers - 2:
                    _, x, feature_map = self.cells[i](dense_feature_map[:-1], x)
                    dense_feature_map.append(feature_map)
                elif i == self.num_model_2_layers -1:
                    x = self.cells[i](dense_feature_map, x)
                else:
                    x = self.cells[i](dense_feature_map[:-1], x)

            if self.args.confidence_map:
                x = self.aspp(x, confidence_map)
            else:
                x = self.aspp(x)
            x = self.decoder(x, low_level, size)     

            torch.cuda.synchronize()
            tic_2 = time.perf_counter()

            return y1, x, tic_1 - tic, tic_2 - tic

        else:
            low_level, dense_feature_map, x = self.model_1(x)
            low_level = self.low_level_conv(low_level)

            y1 = self.aspp(x)
            y1 = self.decoder(y1, low_level, size)

            confidence_map, entropy = normalized_shannon_entropy(y1)
            if entropy < threshold:
                return y1

            for i in range(self.num_model_2_layers):
                if i < self.num_model_2_layers - 2:
                    _, x, feature_map = self.cells[i](dense_feature_map[:-1], x)
                    dense_feature_map.append(feature_map)
                elif i == self.num_model_2_layers -1:
                    x = self.cells[i](dense_feature_map, x)
                else:
                    x = self.cells[i](dense_feature_map[:-1], x)
            if self.args.confidence_map:
                x = self.aspp(x, confidence_map)
            else:
                x = self.aspp(x)
            x = self.decoder(x, low_level, size)     

            return x


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

