import torch
import torch.nn as nn
import numpy as np
from modeling import cell_level_search
from modeling.genotypes import PRIMITIVES
from modeling.genotypes import Genotype
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
import numpy as np
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
                downup_sample):

        super(Cell, self).__init__()
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


    def scale_dimension(self, dim, scale)
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
        s0 = F.interpolate(s0, [s1.shape[2], s1.shape[3]], mode='bilinear') 
            if s0.shape[2] != s1.shape[2] else s0
        s0 = self.pre_preprocess(s0)
        
        states = [s0, s1]
        del s0
        del s1

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


class Model_1 (nn.Module):
    def __init__(self,
                network_arch,
                cell_arch,
                num_classes,
                num_layers,
                BatchNorm,
                F=20,
                B=4,
                low_level_layer=1):

        super(new_device_Model, self).__init__()
        
        self.cells = nn.ModuleList()
        self.model_1_network = network_arch
        self.cell_arch = torch.from_numpy(cell_arch)
        self.num_model_1_layers = num_layers
        self._num_classes = num_classes
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

        self.low_level_conv = nn.Sequential(nn.Conv2d(FB * 2**self.num_model_1_layers.[low_level_layer], 48, 1),
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
                _cell = Cell(BatchNorm,
                            B,
                            64,
                            128,                              
                            self.cell_arch,
                            self.model_1_network[i],
                            F * fm[level],                        
                            downup_sample,
                            pre_downup_sample) 
                
            elif i == 1:
                pre_downup_sample = int(0 - level)
                _cell = Cell(BatchNorm,
                            B,
                            128,
                            FB * fm[prev_level],
                            self.cell_arch,
                            self.model_1_network[i],
                            F * fm[level],
                            downup_sample,
                            pre_downup_sample)
            else:
                _cell = Cell(BatchNorm,
                            B, 
                            FB * fm[prev_prev_level],
                            FB * fm[prev_level],
                            self.cell_arch,
                            self.model_1_network[i],
                            F * fm[level],
                            downup_sample,
                            pre_downup_sample)
       
            self.cells += [_cell]

        if self.model_1_network[-1] == 1:
            mult = 2
        elif self.model_1_network[-1] == 2:
            mult =1
        elif self.model_1_network[-1] ==3:
            mult = 0.5
        else:
            return
        self.aspp_2 = ASPP_train(FB * fm[self.model_1_network[-1]], \
                                      256, num_classes, BatchNorm, mult=mult)
        self._init_weight()

    def forward(self, x):
        stem = self.stem0(x)
        stem0 = self.stem1(stem)
        stem1 = self.stem2(stem0)
        two_last_inputs = (stem0, stem1)

        for i in range(self.num_model_1_layers):       
            two_last_inputs = self.cells[i](
                two_last_inputs[0], two_last_inputs[1])

            if i == self.low_level_layer:
                low_level = two_last_inputs[1]
                low_level = self.low_level_conv(low_level)
            if i == 0:
                del stem
            elif i == 1:
                del stem0
            elif i==2:
                del stem1

        return low_level, two_last_inputs, self.aspp_1(two_last_inputs[-1])

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
    def __init__(self, network_arch, cell_arch_1, cell_arch_2, num_classes, args, low_level_layer):
        super(Model_2, self).__init__()
        BatchNorm = SynchronizedBatchNorm2d if args.sync_bn == True else nn.BatchNorm2d
        F_2 = args.F_2
        F_1 = args.F_1
        B_2 = args.B_2
        B_1 = args.B_1
        num_model_1_layers = args.num_model_1_layers
        self.num_model_2_layers = len(network_arch) - len(num_model_1_layers)
        self.cells = nn.ModuleList()
        self.model_2_network = network_arch[num_model_1_layers:]
        self.cell_arch_2 = torch.from_numpy(cell_arch_2)
        self._num_classes = num_classes

        model_1_network = network_arch[:args.num_model_1_layers]
        self.model_1 = Model_1(model_1_network, cell_arch_1, num_classes, num_model_1_layers, \
                                       BatchNorm, F=F_1, B=B_1, low_level_layer=low_level_layer)
        self.decoder_2 = Decoder(num_classes, BatchNorm)
        
   
        fm = {0: 1, 1: 2, 2: 4, 3: 8}
        for i in range(self.num_model_2_layers):

            level = self.model_2_network[i]
            prev_level = self.model_2_network[i-1]
            prev_prev_level = self.model_2_network[i-2]

            downup_sample = int(prev_level - level)
            pre_downup_sample = int(prev_prev_level - level)

            if i == 0:
                downup_sample = int(model_1_network[-1] - self.model_2_network[0])
                pre_downup_sample = int(model_1_network[-2] - self.model_2_network[0])
                _cell = Cell(BatchNorm, B_2, 
                            F_1 * B_1 * fm[model_1_network[-2]],
                            F_1 * B_1 * fm[model_1_network[-1]],
                            self.cell_arch_2,
                            self.model_2_network[i],
                            F_2 * fm[level],
                            downup_sample,
                            pre_downup_sample)
            
            elif i == 1:
                pre_downup_sample = int(model_1_network[-1] - self.model_2_network[1])
                _cell = Cell(BatchNorm,
                            B_c,
                            F_d * B_d * fm[model_1_network[-1]],
                            F_c * B_c * fm[self.model_2_network[0]],
                            self.cell_arch_2,
                            self.model_2_network[i],
                            F_2 * fm[level],
                            downup_sample,
                            pre_downup_sample)
            else:
                _cell = Cell(BatchNorm, B_c, 
                            F_2 * B_2 * fm[prev_prev_level],
                            F_2 * B_2 * fm[prev_level],
                            self.cell_arch_2,
                            self.model_2_network[i],
                            F_2 * fm[level],
                            downup_sample,
                            pre_downup_sample)

            self.cells += [_cell]

        if self.num_model_2_layers[-1] == 1:
            mult = 2
        elif self.num_model_2_layers[-1] == 2:
            mult =1


        self.aspp_2 = ASPP_train(F_2 * B_2 * fm[self.num_model_2_layers[-1]], 
                                     256, num_classes, BatchNorm, mult=mult)
        self._init_weight()

    def forward(self, x, evaluation=False):
        size = (x.shape[2], x.shape[3])
        if not evaluation:
            low_level, two_last_inputs, y1 = self.model_1(x)
            del x
            for i in range(self.num_model_2_layers):
                two_last_inputs = self.cells[i](
                    two_last_inputs[0], two_last_inputs[1])
                
            y2 = two_last_inputs[-1]
            del two_last_inputs

            y1 = F.interpolate(y1, (low_level.shape[2],low_level.shape[3]), mode='bilinear')
            y1 = self.model_1.decoder_1(y1, low_level, size)

            y2 = self.aspp_cloud(y2)

            y2 = F.interpolate(y2, (low_level.shape[2],low_level.shape[3]), mode='bilinear')
            y2 = self.decoder(y2, low_level, size)     
            del low_level    

            return y1, y2

        else:
            torch.cuda.synchronize()
            tic = time.perf_counter()

            low_level, two_last_inputs, y1 = self.device(x)
            del x
            y1 = F.interpolate(y1, (low_level.shape[2],low_level.shape[3]), mode='bilinear')
            y1 = self.device.decoder(y1, low_level, size)

            torch.cuda.synchronize()
            tic_1 = time.perf_counter()

            for i in range(self.num_model_2_layers):
                two_last_inputs = self.cells[i](
                    two_last_inputs[0], two_last_inputs[1])

            y2 = two_last_inputs[-1]
            y2 = self.aspp_2(y2)
            y2 = F.interpolate(y2, (low_level.shape[2],low_level.shape[3]), mode='bilinear')
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

class ASPP_train(nn.Module):
    def __init__(self, C, depth, num_classes, BatchNorm, conv=nn.Conv2d, eps=1e-5, momentum=0.1, mult=1):
        super(ASPP_train, self).__init__()
        self._C = C
        self._depth = depth
        self._num_classes = num_classes

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.aspp1 = conv(C, depth, kernel_size=1, stride=1, bias=False)
        self.aspp2 = conv(C, depth, kernel_size=3, stride=1,
                               dilation=int(6*mult), padding=int(6*mult),
                               bias=False)
        self.aspp3 = conv(C, depth, kernel_size=3, stride=1,
                               dilation=int(12*mult), padding=int(12*mult),
                               bias=False)
        self.aspp4 = conv(C, depth, kernel_size=3, stride=1,
                               dilation=int(18*mult), padding=int(18*mult),
                               bias=False)
        self.aspp5 = conv(C, depth, kernel_size=1, stride=1, bias=False)
        self.aspp1_bn = BatchNorm(depth)
        self.aspp2_bn = BatchNorm(depth)
        self.aspp3_bn = BatchNorm(depth)
        self.aspp4_bn = BatchNorm(depth)
        self.aspp5_bn = BatchNorm(depth)
        self.conv1 = conv(depth * 5, depth, kernel_size=1, stride=1,
                               bias=False)
        self.bn1 = BatchNorm(depth)
        self._init_weight()

    def forward(self, x):
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
        del x1
        del x2
        del x3
        del x4
        del x5

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
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

class Decoder(nn.Module):

    def __init__(self, n_class, BatchNorm, output_stride=8):
        super(Decoder, self).__init__()
        self.output_stride = output_stride
        self._conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                    BatchNorm(256),
                                    nn.ReLU(),
                                    # 3x3 conv to refine the features
                                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                    BatchNorm(256),
                                    nn.ReLU(),
                                    nn.Conv2d(256, n_class, kernel_size=1, stride=1))
        self._init_weight()

    def forward(self, x, low_level, size):
        x = torch.cat((x, low_level), 1)
        x = self._conv(x)
        x = F.interpolate(x, size, mode='bilinear')

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