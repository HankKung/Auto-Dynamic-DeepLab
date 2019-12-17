import torch
import torch.nn as nn
import numpy as np
import cell_level_search
from genotypes import PRIMITIVES
from genotypes import Genotype
import torch.nn.functional as F
import numpy as np
from operations import *
from modeling.decoder import *





class Cell(nn.Module):

    def __init__(self, steps, block_multiplier, prev_prev_fmultiplier,
                 prev_filter_multiplier,
                 cell_arch, network_arch,
                 filter_multiplier, downup_sample, pre_downup_sample, prev_prev_block=0):

        super(Cell, self).__init__()
        eps = 1e-5
        momentum = 0.1
        self.cell_arch = cell_arch
        self.C_in = block_multiplier * filter_multiplier
        self.C_out = filter_multiplier
        self.C_prev = int(block_multiplier * prev_filter_multiplier)
        if prev_prev_block != 0:
            self.C_prev_prev = int(prev_prev_block * prev_prev_fmultiplier)
        else:
            self.C_prev_prev = int(block_multiplier * prev_prev_fmultiplier)
        self.downup_sample = downup_sample
        self.pre_downup_sample = pre_downup_sample
        self.pre_preprocess = ReLUConvBN(
            self.C_prev_prev, self.C_out, 1, 1, 0, eps=eps, momentum=momentum, affine=True)
        self.preprocess = ReLUConvBN(
            self.C_prev, self.C_out, 1, 1, 0, eps=eps, momentum=momentum, affine=True)
        self._steps = steps
        self.block_multiplier = block_multiplier
        self._ops = nn.ModuleList()
        if downup_sample == -1:
            self.preprocess = FactorizedReduce(self.C_prev, self.C_out, eps=eps, momentum=momentum)
        elif downup_sample == 1:
            self.scale = 2

        if pre_downup_sample == -1:
            self.pre_preprocess = FactorizedReduce(self.C_prev_prev, self.C_out, eps=eps, momentum=momentum)
        elif pre_downup_sample == -2:
            self.pre_preprocess = DoubleFactorizedReduce(self.C_prev_prev, self.C_out, eps=eps, momentum=momentum)
        elif pre_downup_sample == 1:
            self.pre_pre_scale = 2
        elif pre_downup_sample == 2:
            self.pre_pre_scale = 4

         
        for x in self.cell_arch:
            primitive = PRIMITIVES[x[1]]
            op = OPS[primitive](self.C_out, stride=1, eps=eps, momentum=momentum, affine=True)
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
        s0 = F.interpolate(s0, \
            (self.scale_dimension(s0.shape[2], self.pre_pre_scale), \
            self.scale_dimension(s0.shape[3], self.pre_pre_scale)), \
            mode='bilinear') if self.pre_downup_sample > 0 else s0

        s0 = self.pre_preprocess(s0)
        
        states = [s0, s1]

        offset = 0
        ops_index = 0
        for i in range(self._steps):
            new_states = []
            for j, h in enumerate(states):
                branch_index = offset + j
                if branch_index in self.cell_arch[:, 0]:
                    if prev_prev_input is None and j == 0:
                        ops_index += 1
                        continue
                    new_state = self._ops[ops_index](h)
                    new_states.append(new_state)
                    ops_index += 1

            s = sum(new_states)
            offset += len(states)
            states.append(s)

        concat_feature = torch.cat(states[-self._steps:], dim=1)
        return prev_input, concat_feature


class new_device_Model (nn.Module):
    def __init__(self, network_arch, cell_arch, num_classes, num_layers, \
        criterion=None, filter_multiplier=20, block_multiplier=4, step=4, \
        cell=Cell, full_net='deeplab_v3+'):
        super(new_device_Model, self).__init__()
        
        self.cells = nn.ModuleList()
        self.network_arch = network_arch
        self.cell_arch = torch.from_numpy(cell_arch)
        self._num_layers = num_layers
        self._num_classes = num_classes
        self._step = step
        self._block_multiplier = block_multiplier
        self._filter_multiplier = filter_multiplier
        self._criterion = criterion
        self._full_net = full_net
        initial_fm = 128
        self.stem0 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.stem1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # TODO: first two channels should be set automatically
        ini_initial_fm = 64
        self.stem2 = nn.Sequential(
            nn.Conv2d(64, initial_fm, 3, stride=2, padding=1),
            nn.BatchNorm2d(initial_fm),
            nn.ReLU()
        )
       
        filter_param_dict = {0: 1, 1: 2, 2: 4, 3: 8}
        for i in range(self._num_layers):

            level = network_arch[i]
            prev_level = network_arch[i-1]
            prev_prev_level = network_arch[i-2]

            if i == 0:
                downup_sample = int(0 - level)
                pre_downup_sample = int(-1 - level)
                _cell = cell(self._step, self._block_multiplier, ini_initial_fm / block_multiplier,
                             initial_fm / block_multiplier,
                             self.cell_arch, self.network_arch[i],
                             self._filter_multiplier *
                             filter_param_dict[level],
                             downup_sample,
                             pre_downup_sample)
            else:

                downup_sample = int(prev_level- level)
                
                if i == 1:
                    pre_downup_sample = int(0 - level)
                    _cell = cell(self._step, self._block_multiplier,
                                 initial_fm / block_multiplier,
                                 self._filter_multiplier * filter_param_dict[prev_level],
                                 self.cell_arch, self.network_arch[i],
                                 self._filter_multiplier *
                                 filter_param_dict[level],
                                 downup_sample,
                                 pre_downup_sample)
                else:
                    downup_sample = int(prev_level - level)
                    pre_downup_sample = int(prev_prev_level - level)
                    _cell = cell(self._step, self._block_multiplier, self._filter_multiplier * filter_param_dict[prev_prev_level],
                                 self._filter_multiplier *
                                 filter_param_dict[prev_level],
                                 self.cell_arch, self.network_arch[i],
                                 self._filter_multiplier *
                                 filter_param_dict[level],
                                 downup_sample,
                                 pre_downup_sample)
       
            self.cells += [_cell]
        if network_arch[-1] == 1:
            mult = 2
        elif network_arch[-1] == 2:
            mult =1
        elif network_arch[-1] ==3:
            mult = 0.5
        else:
            return
        self.aspp_device = ASPP_train(filter_multiplier * step * filter_param_dict[self.network_arch[-1]], \
                                      256, \
                                      num_classes, mult=mult)

    def forward(self, x):
        stem = self.stem0(x)
        stem0 = self.stem1(stem)
        stem1 = self.stem2(stem0)

        two_last_inputs = (stem0, stem1)
        for i in range(self._num_layers):       
            two_last_inputs = self.cells[i](
                two_last_inputs[0], two_last_inputs[1])
   
        last_output = two_last_inputs[-1]
        aspp_result = self.aspp_device(last_output)
        return  two_last_inputs, aspp_result

class new_cloud_Model (nn.Module):
    def __init__(self, network_arch, cell_arch_d, cell_arch_c, num_classes, \
        device_num_layers, \
        criterion=None, filter_multiplier=20, block_multiplier_c=5, step_c=5, \
        block_multiplier_d=4, step_d=4, cell=Cell, full_net='deeplab_v3+'):
        super(new_cloud_Model, self).__init__()

        self.cells = nn.ModuleList()
        self.network_arch = network_arch[device_num_layers:]
        self.cell_arch_c = torch.from_numpy(cell_arch_c)
        self._num_layers = len(self.network_arch)
        self._num_classes = num_classes
        self._step = step_c
        self._block_multiplier_c = block_multiplier_c
        self._block_multiplier_d = block_multiplier_d
        self._filter_multiplier = filter_multiplier
        self._criterion = criterion
        self._full_net = full_net

        device_layer = network_arch[:device_num_layers]
        self.device = new_device_Model(device_layer, cell_arch_d, num_classes, device_num_layers \
            ,block_multiplier=block_multiplier_d, step=step_d)
        self._num_layers = 12 - len(device_layer)     
   
        filter_param_dict = {0: 1, 1: 2, 2: 4, 3: 8}
        for i in range(self._num_layers):

            level = self.network_arch[i]
            prev_level = self.network_arch[i-1]
            prev_prev_level = self.network_arch[i-2]

            if i == 0:
                downup_sample = int(device_layer[-1]-self.network_arch[0])
                pre_downup_sample = int(device_layer[-2] - self.network_arch[0])
                _cell = cell(self._step, self._block_multiplier_d, self._filter_multiplier * filter_param_dict[device_layer[-2]],
                             self._filter_multiplier * filter_param_dict[device_layer[-1]],
                             self.cell_arch_c, self.network_arch[i],
                             self._filter_multiplier *
                             filter_param_dict[level],
                             downup_sample,
                             pre_downup_sample)
            else:
                downup_sample = int(prev_level - level)
                if i == 1:
                    pre_downup_sample = int(device_layer[-1] - self.network_arch[1])
                    _cell = cell(self._step, self._block_multiplier_c,
                                 self._filter_multiplier * filter_param_dict[device_layer[-1]],
                                 self._filter_multiplier * filter_param_dict[self.network_arch[0]],
                                 self.cell_arch_c, self.network_arch[i],
                                 self._filter_multiplier *
                                 filter_param_dict[level],
                                 downup_sample,
                                 pre_downup_sample,
                                 prev_prev_block=step_d)
                else:
                    downup_sample = int(prev_level - level)
                    pre_downup_sample = int(prev_prev_level - level)
                    _cell = cell(self._step, self._block_multiplier_c, self._filter_multiplier * filter_param_dict[prev_prev_level],
                                 self._filter_multiplier *
                                 filter_param_dict[prev_level],
                                 self.cell_arch_c, self.network_arch[i],
                                 self._filter_multiplier *
                                 filter_param_dict[level],
                                 downup_sample,
                                 pre_downup_sample)
            print(downup_sample)          
            self.cells += [_cell]

        if network_arch[-1] == 1:
            mult = 2
        elif network_arch[-1] == 2:
            mult =1
        elif network_arch[-1] == 3:
            mult = 0.5
        else:
            return

        self.aspp_cloud = ASPP_train(filter_multiplier * step_c * filter_param_dict[self.network_arch[-1]], \
                                     256, \
                                     num_classes, mult=mult)

    def forward(self, x):
        size = (x.shape[2], x.shape[3])
        two_last_inputs, device_output = self.device(x)
        for i in range(self._num_layers):
            two_last_inputs = self.cells[i](
                two_last_inputs[0], two_last_inputs[1])
            
        last_output = two_last_inputs[-1]
        device_output = F.interpolate(device_output, size, mode='bilinear')
        cloud_output = self.aspp_cloud(last_output)
        cloud_output = F.interpolate(cloud_output, size, mode='bilinear')           

        return device_output, cloud_output

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
        modules = [self.device.aspp_device, self.aspp_cloud]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d)or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

class ASPP_train(nn.Module):
    def __init__(self, C, depth, num_classes, conv=nn.Conv2d, norm=nn.BatchNorm2d, eps=1e-5, momentum=0.1, mult=1):
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
        self.aspp1_bn = norm(depth)
        self.aspp2_bn = norm(depth)
        self.aspp3_bn = norm(depth)
        self.aspp4_bn = norm(depth)
        self.aspp5_bn = norm(depth)
        self.conv2 = conv(depth * 5, depth, kernel_size=1, stride=1,
                               bias=False)
        self.bn2 = norm(depth)
        self.conv3 = nn.Conv2d(depth, num_classes, kernel_size=1, stride=1)

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
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)

        return x

def network_layer_to_space(net_arch):
    for i, layer in enumerate(net_arch):
        if i == 0:
            space = np.zeros((1, 4, 3))
            space[0][layer][0] = 1
            prev = layer
        else:
            if layer == prev + 1:
                sample = 0
            elif layer == prev:
                sample = 1
            elif layer == prev - 1:
                sample = 2
            space1 = np.zeros((1, 4, 3))
            space1[0][layer][sample] = 1
            space = np.concatenate([space, space1], axis=0)
            prev = layer
    return space


def get_cell():
    cell = np.zeros((10, 2))
    cell[0] = [0, 7]
    cell[1] = [1, 4]
    cell[2] = [2, 4]
    cell[3] = [3, 6]
    cell[4] = [5, 4]
    cell[5] = [8, 4]
    cell[6] = [11, 5]
    cell[7] = [13, 5]
    cell[8] = [19, 7]
    cell[9] = [18, 5]
    return cell.astype('uint8'), cell.astype('uint8')


def get_arch():

    backbone = [0, 0, 0, 1, 2, 1, 2, 2, 3, 3, 2, 1]
    network_arch = network_layer_to_space(backbone)
    cell_arch, cell_arch2 = get_cell()

    return network_arch, cell_arch, cell_arch2


def get_default_net(filter_multiplier=8):
    net_arch, cell_arch, cell_arch2 = get_arch()
    return new_cloud_Model(net_arch, cell_arch, cell_arch2, 19, 6, filter_multiplier=filter_multiplier)
 
