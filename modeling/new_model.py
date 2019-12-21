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

    def __init__(self, BatchNorm, B, 
                 prev_prev_C, prev_C, 
                 cell_arch, network_arch,
                 C_out,
                 downup_sample, pre_downup_sample):

        super(Cell, self).__init__()
        eps = 1e-5
        momentum = 0.1

        self.cell_arch = cell_arch
        self.downup_sample = downup_sample
        self.pre_downup_sample = pre_downup_sample
        self.pre_preprocess = ReLUConvBN(
            prev_prev_C, C_out, 1, 1, 0, BatchNorm, eps=eps, momentum=momentum, affine=True)
        self.preprocess = ReLUConvBN(
            prev_C, C_out, 1, 1, 0, BatchNorm, eps=eps, momentum=momentum, affine=True)
        self.B = B
        self._ops = nn.ModuleList()
        if downup_sample == -1:
            self.preprocess = FactorizedReduce(prev_C, C_out, BatchNorm, eps=eps, momentum=momentum)
        elif downup_sample == 1:
            self.scale = 2

        if pre_downup_sample == -1:
            self.pre_preprocess = FactorizedReduce(prev_prev_C, C_out, BatchNorm, eps=eps, momentum=momentum)
        elif pre_downup_sample == -2:
            self.pre_preprocess = DoubleFactorizedReduce(prev_prev_C, C_out, BatchNorm, eps=eps, momentum=momentum)
        elif pre_downup_sample == 1:
            self.pre_pre_scale = 2
        elif pre_downup_sample == 2:
            self.pre_pre_scale = 4

         
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
        s0 = F.interpolate(s0, \
            (self.scale_dimension(s0.shape[2], self.pre_pre_scale), \
            self.scale_dimension(s0.shape[3], self.pre_pre_scale)), \
            mode='bilinear') if self.pre_downup_sample > 0 else s0

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

class new_device_Model (nn.Module):
    def __init__(self, network_arch, cell_arch, num_classes, num_layers, BatchNorm,\
        criterion=None, F=20, B=4):
        super(new_device_Model, self).__init__()
        
        self.cells = nn.ModuleList()
        self.network_arch = network_arch
        self.cell_arch = torch.from_numpy(cell_arch)
        self._num_layers = num_layers
        self._num_classes = num_classes

        self.low_level_conv = nn.Sequential(nn.Conv2d(160, 48, 1),
                                    BatchNorm(48),
                                    nn.ReLU())
        self.decoder = Decoder(num_classes, BatchNorm)

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            BatchNorm(64),
            nn.ReLU()
        )
        self.stem1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            BatchNorm(64),
            nn.ReLU()
        )

        self.stem2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            BatchNorm(128),
            nn.ReLU()
        )
        FB = F * B
        fm = {0: 1, 1: 2, 2: 4, 3: 8}
        for i in range(self._num_layers):

            level = network_arch[i]
            prev_level = network_arch[i-1]
            prev_prev_level = network_arch[i-2]

            downup_sample = int(prev_level - level)
            pre_downup_sample = int(prev_prev_level - level)
            if i == 0:
                downup_sample = int(0 - level)
                pre_downup_sample = int(-1 - level)
                _cell = Cell(BatchNorm, B,
                             64, 128,                               # pre_pre_c & pre_c
                             self.cell_arch, self.network_arch[i],
                             F * fm[level],                         # C_out
                             downup_sample, pre_downup_sample) 
                
            elif i == 1:
                pre_downup_sample = int(0 - level)
                _cell = Cell(BatchNorm, B,
                             128, FB * fm[prev_level],
                             self.cell_arch, self.network_arch[i],
                             F * fm[level],
                             downup_sample, pre_downup_sample)
            else:
                _cell = Cell(BatchNorm, B, 
                             FB * fm[prev_prev_level], FB * fm[prev_level],
                             self.cell_arch, self.network_arch[i],
                             F * fm[level],
                             downup_sample, pre_downup_sample)
       
            self.cells += [_cell]

        if network_arch[-1] == 1:
            mult = 2
        elif network_arch[-1] == 2:
            mult =1
        elif network_arch[-1] ==3:
            mult = 0.5
        else:
            return
        self.aspp_device = ASPP_train(FB * fm[self.network_arch[-1]], \
                                      256, num_classes, BatchNorm, mult=mult)
        self._init_weight()

    def forward(self, x):
        stem = self.stem0(x)
        stem0 = self.stem1(stem)
        stem1 = self.stem2(stem0)

        two_last_inputs = (stem0, stem1)
        for i in range(self._num_layers):       
            two_last_inputs = self.cells[i](
                two_last_inputs[0], two_last_inputs[1])
            if i == 0:
                del stem
            elif i == 1:
                low_level = two_last_inputs[1]
                low_level = self.low_level_conv(low_level)
                del stem0
            elif i==2:
                del stem1
   
        last_output = two_last_inputs[-1]
        aspp_result = self.aspp_device(last_output)
        return low_level, two_last_inputs, aspp_result

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class new_cloud_Model (nn.Module):
    def __init__(self, network_arch, cell_arch_d, cell_arch_c, num_classes, \
        device_num_layers, F=20, B_c=5, \
        B_d=4, sync_bn=False):
        super(new_cloud_Model, self).__init__()

        BatchNorm = SynchronizedBatchNorm2d if sync_bn == True else nn.BatchNorm2d
        self.cells = nn.ModuleList()
        self.network_arch = network_arch[device_num_layers:]
        self.cell_arch_c = torch.from_numpy(cell_arch_c)
        self._num_layers = len(self.network_arch)
        self._num_classes = num_classes

        device_layer = network_arch[:device_num_layers]
        self.device = new_device_Model(device_layer, cell_arch_d, num_classes, device_num_layers, \
                                       BatchNorm, B=B_d)
        self.decoder = Decoder(num_classes, BatchNorm)
        self._num_layers = 12 - len(device_layer)     
   
        fm = {0: 1, 1: 2, 2: 4, 3: 8}
        for i in range(self._num_layers):

            level = self.network_arch[i]
            prev_level = self.network_arch[i-1]
            prev_prev_level = self.network_arch[i-2]

            downup_sample = int(prev_level - level)
            pre_downup_sample = int(prev_prev_level - level)

            if i == 0:
                downup_sample = int(device_layer[-1]-self.network_arch[0])
                pre_downup_sample = int(device_layer[-2] - self.network_arch[0])
                _cell = Cell(BatchNorm, B_c, 
                             F * B_d * fm[device_layer[-2]], F * B_d * fm[device_layer[-1]],
                             self.cell_arch_c, self.network_arch[i],
                             F * fm[level],
                             downup_sample, pre_downup_sample)
            
            elif i == 1:
                pre_downup_sample = int(device_layer[-1] - self.network_arch[1])
                _cell = Cell(BatchNorm, B_c,
                             F * B_d * fm[device_layer[-1]], F * B_c * fm[self.network_arch[0]],
                             self.cell_arch_c, self.network_arch[i],
                             F * fm[level],
                             downup_sample, pre_downup_sample)
            else:
                _cell = Cell(BatchNorm, B_c, 
                             F * B_c * fm[prev_prev_level], F * B_c * fm[prev_level],
                             self.cell_arch_c, self.network_arch[i],
                             F * fm[level],
                             downup_sample, pre_downup_sample)
            self.cells += [_cell]

        if network_arch[-1] == 1:
            mult = 2
        elif network_arch[-1] == 2:
            mult =1
        elif network_arch[-1] == 3:
            mult = 0.5
        else:
            return

        self.aspp_cloud = ASPP_train(F * B_c * fm[self.network_arch[-1]], 
                                     256, num_classes, BatchNorm, mult=mult)
        self._init_weight()

    def forward(self, x, evaluation=False):
        size = (x.shape[2], x.shape[3])
        if not evaluation:
            low_level, two_last_inputs, device_output = self.device(x)
            for i in range(self._num_layers):
                two_last_inputs = self.cells[i](
                    two_last_inputs[0], two_last_inputs[1])
                
            last_output = two_last_inputs[-1]
            del two_last_inputs

            device_output = F.interpolate(device_output, (low_level.shape[2],low_level.shape[3]), mode='bilinear')
            device_output = self.device.decoder(device_output, low_level, size)

            cloud_output = self.aspp_cloud(last_output)
            del last_output
            cloud_output = F.interpolate(cloud_output, (low_level.shape[2],low_level.shape[3]), mode='bilinear')
            cloud_output = self.decoder(cloud_output, low_level, size)         

            return device_output, cloud_output

        else:
            torch.cuda.synchronize()
            tic = time.perf_counter()

            low_level, two_last_inputs, device_output = self.device(x)
            device_output = F.interpolate(device_output, (low_level.shape[2],low_level.shape[3]), mode='bilinear')
            device_output = self.device.decoder(device_output, low_level, size)

            torch.cuda.synchronize()
            tic_1 = time.perf_counter()

            for i in range(self._num_layers):
                two_last_inputs = self.cells[i](
                    two_last_inputs[0], two_last_inputs[1])
            cloud_output = self.aspp_cloud(last_output)
            cloud_output = F.interpolate(cloud_output, (low_level.shape[2],low_level.shape[3]), mode='bilinear')
            cloud_output = self.decoder(cloud_output, low_level, size)

            torch.cuda.synchronize()
            tic_2 = time.perf_counter()

            return device_output, cloud_output, tic_1 - tic, tic_2 - tic

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

    def get_1x_lr_params(self):
        modules = [self.device.stem0, self.device.stem1, self.device.stem2, self.device.cells, self.cells]
        for i in range(len(modules)):
            if i < 3:
                for m in modules[i].named_modules():
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
            else:
                for cell in modules[i]:
                    for m in cell.named_modules():
                        if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                            for p in m[1].parameters():
                                if p.requires_grad:
                                    yield p
    def get_10x_lr_params(self):
        modules = [self.device.aspp_device, self.aspp_cloud, self.device.decoder, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d)or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

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
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
