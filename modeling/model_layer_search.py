import torch
import torch.nn as nn
import numpy as np
from modeling.genotypes import PRIMITIVES
import torch.nn.functional as F
from modeling.operations import *
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class Cell(nn.Module):

    def __init__(self,
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
        eps = 1e-3
        momentum = 3e-4
        BatchNor = nn.BatchNorm2d
        self.cell_arch = cell_arch
        self.downup_sample = downup_sample
        self.B = B
        self.dense_in = dense_in
        self.dense_out = dense_out


        if prev_C_down is not None:  
            self.preprocess_down = FactorizedReduce(
                prev_C_down, C_out, BatchNorm=BatchNorm, affine=False)
        if prev_C_same is not None:
            self.preprocess_same = ReLUConvBN(
                prev_C_same, C_out, 1, 1, 0, BatchNorm=BatchNorm, affine=False)
        if prev_C_up is not None:
            self.preprocess_up = ReLUConvBN(
                prev_C_up, C_out, 1, 1, 0, BatchNorm=BatchNorm, affine=False)


        self._ops = nn.ModuleList()
        if prev_prev_C != -1:
            if pre_preprocess_sample_rate >= 1:
                self.pre_preprocess = ReLUConvBN(
                    prev_prev_C, C_out, 1, 1, 0, BatchNorm=BatchNorm, affine=False)
            elif pre_preprocess_sample_rate == 0.5:
                self.pre_preprocess = FactorizedReduce(
                    prev_prev_C, C_out, BatchNorm=BatchNorm, affine=False)
            elif pre_preprocess_sample_rate == 0.25:
                self.pre_preprocess = DoubleFactorizedReduce(
                    prev_prev_C, C_out, BatchNorm=BatchNorm, affine=False)

        for x in self.cell_arch:
            primitive = PRIMITIVES[x[1]]
            op = OPS[primitive](C_out, 1, BatchNorm, eps=eps, momentum=momentum, affine=False)
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


  def forward(self, x)
class Model_layer_search (nn.Module) :
    def __init__(self,
                num_classes,
                num_layers,
                F=8,
                B_1=5,
                B_2=5, 
                exit_layer=5,
                sync_bn=False,
                cell=cell_level_search.Cell):

        super(Model_layer_search, self).__init__()

        BatchNorm = SynchronizedBatchNorm2d if sync_bn == True else nn.BatchNorm2d
        self.cells = nn.ModuleList()
        self._num_layers = num_layers
        self._num_classes = num_classes
        self.B_1 = B_1
        self.B_2 = B_2
        self.exit_layer = exit_layer
        self._initialize_alphas_betas ()

        f_initial = F * B_1
        half_f_initial = int(f_initial / 2)

        FB1 = F * B_1
        FB2 = F * B_2

        self.dense_preprocess = nn.ModuleList()
        for i in range(self._num_layers-2):
            if i == 0:
                self.dense_preprocess.append(nn.ModuleList())
                self.dense_preprocess[0].append(ReLUConvBN(FB1, F, 1, 1, 0, BatchNorm=BatchNorm, affine=False))
                self.dense_preprocess[0].append(ReLUConvBN(FB1 * 2, F * 2, 1, 1, 0, BatchNorm=BatchNorm, affine=False))
                self.dense_preprocess[0].append(FactorizedReduce(FB1 * 2, F * 4, BatchNorm=BatchNorm, affine=False))
                self.dense_preprocess[0].append(DoubleFactorizedReduce(FB1 * 2, F * 8, BatchNorm=BatchNorm, affine=False))
            elif i == 1:
                self.dense_preprocess.append(nn.ModuleList())
                self.dense_preprocess[1].append(ReLUConvBN(FB1, F, 1, 1, 0, BatchNorm=BatchNorm, affine=False))
                self.dense_preprocess[1].append(ReLUConvBN(FB1 * 2, F * 2, 1, 1, 0, BatchNorm=BatchNorm, affine=False))
                self.dense_preprocess[1].append(ReLUConvBN(FB1 * 4, F * 4, 1, 1, 0, BatchNorm=BatchNorm, affine=False))
                self.dense_preprocess[1].append(FactorizedReduce(FB1 * 4, F * 8, BatchNorm=BatchNorm, affine=False))
            elif i <= self.exit_layer:
                self.dense_preprocess.append(nn.ModuleList())
                self.dense_preprocess[i].append(ReLUConvBN(FB1, F, 1, 1, 0, BatchNorm=BatchNorm, affine=False))
                self.dense_preprocess[i].append(ReLUConvBN(FB1 * 2, F * 2, 1, 1, 0, BatchNorm=BatchNorm, affine=False))
                self.dense_preprocess[i].append(ReLUConvBN(FB1 * 4, F * 4, 1, 1, 0, BatchNorm=BatchNorm, affine=False))
                self.dense_preprocess[i].append(ReLUConvBN(FB1 * 8, F * 8, 1, 1, 0, BatchNorm=BatchNorm, affine=False))
            else:
                self.dense_preprocess.append(nn.ModuleList())
                self.dense_preprocess[i].append(ReLUConvBN(FB2, F, 1, 1, 0, BatchNorm=BatchNorm, affine=False))
                self.dense_preprocess[i].append(ReLUConvBN(FB2 * 2, F * 2, 1, 1, 0, BatchNorm=BatchNorm, affine=False))
                self.dense_preprocess[i].append(ReLUConvBN(FB2 * 4, F * 4, 1, 1, 0, BatchNorm=BatchNorm, affine=False))
                self.dense_preprocess[i].append(ReLUConvBN(FB2 * 8, F * 8, 1, 1, 0, BatchNorm=BatchNorm, affine=False))

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, half_f_initial, 3, stride=2, padding=1),
            BatchNorm(half_f_initial),
        )
        self.stem1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(half_f_initial, f_initial, 3, stride=2, padding=1),
            BatchNorm(f_initial),
        )

        """ build the cells """
        for i in range (self._num_layers):

            if i == 0 :
                cell1 = cell (B_1, half_f_initial,
                              None, f_initial, None,
                              F, BatchNorm=BatchNorm, pre_preprocess_sample_rate=0.5)
                cell2 = cell (B_1, half_f_initial,
                              f_initial, None, None,
                              F * 2, BatchNorm=BatchNorm, pre_preprocess_sample_rate=0.25)
                self.cells += [cell1]
                self.cells += [cell2]
            elif i == 1 :
                cell1 = cell (B_1, f_initial,
                              None, FB1, FB1 * 2,
                              F, BatchNorm=BatchNorm)

                cell2 = cell (B_1, f_initial,
                              FB1, FB1 * 2, None,
                              F * 2, BatchNorm=BatchNorm, pre_preprocess_sample_rate=0.5)

                cell3 = cell (B_1, f_initial,
                              FB1 * 2, None, None,
                              F * 4, BatchNorm=BatchNorm, pre_preprocess_sample_rate=0.25)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]

            elif i == 2 :
                cell1 = cell (B_1, FB1,
                              None, FB1, FB1 * 2,
                              F, BatchNorm=BatchNorm)

                cell2 = cell (B_1, FB1 * 2,
                              FB1, FB1 * 2, FB1 * 4,
                              F * 2, BatchNorm=BatchNorm)

                cell3 = cell (B_1, FB1 * 2,
                              FB1 * 2, FB1 * 4, None,
                              F * 4, BatchNorm=BatchNorm, pre_preprocess_sample_rate=0.5)

                cell4 = cell (B_1, FB1 * 2,
                              FB1 * 4, None, None,
                              F * 8, BatchNorm=BatchNorm, pre_preprocess_sample_rate=0.25)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]

            elif i == 3 :
                cell1 = cell (B_1, F * (i-1),
                              None, FB1, FB1 * 2,
                              F, BatchNorm=BatchNorm)

                cell2 = cell (B_1, F * (i-1) * 2,
                              FB1, FB1 * 2, FB1 * 4,
                              F * 2, BatchNorm=BatchNorm)

                cell3 = cell (B_1, F * (i-1) * 4,
                              FB1 * 2, FB1 * 4, FB1 * 8,
                              F * 4, BatchNorm=BatchNorm)


                cell4 = cell (B_1, F * (i-1) * 8,
                              FB1 * 4, FB1 * 8, None,
                              F * 8, BatchNorm=BatchNorm)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]

            elif i < exit_layer :
                cell1 = cell (B_1, F * (i-1),
                              None, FB1, FB1 * 2,
                              F, BatchNorm=BatchNorm)

                cell2 = cell (B_1, F * (i-1) * 2,
                              FB1, FB1 * 2, FB1 * 4,
                              F * 2, BatchNorm=BatchNorm)

                cell3 = cell (B_1, F * (i-1) * 4,
                              FB1 * 2, FB1 * 4, FB1 * 8,
                              F * 4, BatchNorm=BatchNorm)

                cell4 = cell (B_1, F * (i-1) * 8,
                              FB1 * 4, FB1 * 8, None,
                              F * 8, BatchNorm=BatchNorm)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]

            elif i == exit_layer:
                cell1 = cell (B_1, F * (i-1),
                              None, FB1, FB1 * 2,
                              F, BatchNorm=BatchNorm)

                cell2 = cell (B_1, F * (i-1) * 2,
                              FB1, FB1 * 2, FB1 * 4,
                              F * 2, BatchNorm=BatchNorm)

                cell3 = cell (B_1, F * (i-1) * 4,
                              FB1 * 2, FB1 * 4, FB1 * 8,
                              F * 4, BatchNorm=BatchNorm)

                cell4 = cell (B_1, F * (i-1) * 8,
                              FB1 * 4, FB1 * 8, None,
                              F * 8, BatchNorm=BatchNorm)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]

            elif i == exit_layer+1:
                cell1 = cell (B_2, F * (i-1),
                              None, FB1, FB1 * 2,
                              F, BatchNorm=BatchNorm)

                cell2 = cell (B_2, F * (i-1) * 2,
                              FB1, FB1 * 2, FB1 * 4,
                              F * 2, BatchNorm=BatchNorm)

                cell3 = cell (B_2, F * (i-1) * 4,
                              FB1 * 2, FB1 * 4, FB1 * 8,
                              F * 4, BatchNorm=BatchNorm)

                cell4 = cell (B_2, F * (i-1) * 8,
                              FB1 * 4, FB1 * 8, None,
                              F * 8, BatchNorm=BatchNorm)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]

            else:
                cell1 = cell (B_2, F * (i-1),
                              None, FB2, FB2 * 2,
                              F, BatchNorm=BatchNorm)

                cell2 = cell (B_2, F * (i-1) * 2,
                              FB2, FB2 * 2, FB2 * 4,
                              F * 2, BatchNorm=BatchNorm)

                cell3 = cell (B_2, F * (i-1) * 4,
                              FB2 * 2, FB2 * 4, FB2 * 8,
                              F * 4, BatchNorm=BatchNorm)

                cell4 = cell (B_2, F * (i-1) * 8,
                              FB2 * 4, FB2 * 8, None,
                              F * 8, BatchNorm=BatchNorm)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]

        # self.aspp_exit_1_4 = nn.Sequential (
        #     ASPP (FB1, self._num_classes, 24, 24, BatchNorm=BatchNorm) #96 / 4 as in the paper
        # )
        self.aspp_exit_1_8 = nn.Sequential (
            ASPP (FB1 * 2, self._num_classes, 12, 12, BatchNorm=BatchNorm) #96 / 8
        )
        # self.aspp_exit_1_16 = nn.Sequential (
        #     ASPP (FB1 * 4, self._num_classes, 6, 6, BatchNorm=BatchNorm) #96 / 16
        # )
        # self.aspp_exit_1_32 = nn.Sequential (
        #     ASPP (FB1 * 8, self._num_classes, 3, 3, BatchNorm=BatchNorm) #96 / 32
        # )

        # self.aspp_exit_2_4 = nn.Sequential (
        #     ASPP (FB2, self._num_classes, 24, 24, BatchNorm=BatchNorm) #96 / 4 as in the paper
        # )
        self.aspp_exit_2_8 = nn.Sequential (
            ASPP (FB2 * 2, self._num_classes, 12, 12, BatchNorm=BatchNorm) #96 / 8
        )
        # self.aspp_exit_2_16 = nn.Sequential (
        #     ASPP (FB2 * 4, self._num_classes, 6, 6, BatchNorm=BatchNorm) #96 / 16
        # )
        # self.aspp_exit_2_32 = nn.Sequential (
        #     ASPP (FB2 * 8, self._num_classes, 3, 3, BatchNorm=BatchNorm) #96 / 32
        # )
        self._init_weight()


    def forward (self, x, training=True, alphas_1=None, alphas_2=None, betas=None) :
        level_4 = []
        level_8 = []
        level_16 = []
        level_32 = []

        level_4_dense = []
        level_8_dense = []
        level_16_dense = []
        level_32_dense = []

        temp = self.stem0 (x)
        level_4.append (self.stem1(temp))

        count = 0

        normalized_betas = torch.randn(12, 4, 3).cuda().half()

        """ Softmax on alphas and betas """
        if training:
            if torch.cuda.device_count() > 1:
                img_device = torch.device('cuda', x.get_device())
                normalized_alphas_1 = F.softmax(self.alphas_1.to(device=img_device), dim=-1)
                normalized_alphas_2 = F.softmax(self.alphas_2.to(device=img_device), dim=-1)

                """ normalized_betas[layer][ith node][0 : ➚, 1: ➙, 2 : ➘] """
                for layer in range (len(self.betas)):
                    if layer == 0:
                        normalized_betas[layer][0][1:] = F.softmax (self.betas[layer][0][1:].to(device=img_device), dim=-1) * (2/3)

                    elif layer == 1:
                        normalized_betas[layer][0][1:] = F.softmax (self.betas[layer][0][1:].to(device=img_device), dim=-1) * (2/3)
                        normalized_betas[layer][1] = F.softmax (self.betas[layer][1].to(device=img_device), dim=-1)

                    elif layer == 2:
                        normalized_betas[layer][0][1:] = F.softmax (self.betas[layer][0][1:].to(device=img_device), dim=-1) * (2/3)
                        normalized_betas[layer][1] = F.softmax (self.betas[layer][1].to(device=img_device), dim=-1)
                        normalized_betas[layer][2] = F.softmax (self.betas[layer][2].to(device=img_device), dim=-1)
                    else :
                        normalized_betas[layer][0][1:] = F.softmax (self.betas[layer][0][1:].to(device=img_device), dim=-1) * (2/3)
                        normalized_betas[layer][1] = F.softmax (self.betas[layer][1].to(device=img_device), dim=-1)
                        normalized_betas[layer][2] = F.softmax (self.betas[layer][2].to(device=img_device), dim=-1)
                        normalized_betas[layer][3][:2] = F.softmax (self.betas[layer][3][:2].to(device=img_device), dim=-1) * (2/3)

            else:
                normalized_alphas_1 = F.softmax(self.alphas_1, dim=-1)
                normalized_alphas_2 = F.softmax(self.alphas_2, dim=-1)

                for layer in range (len(self.betas)):
                    if layer == 0:
                        normalized_betas[layer][0][1:] = F.softmax (self.betas[layer][0][1:], dim=-1) * (2/3)

                    elif layer == 1:
                        normalized_betas[layer][0][1:] = F.softmax (self.betas[layer][0][1:], dim=-1) * (2/3)
                        normalized_betas[layer][1] = F.softmax (self.betas[layer][1], dim=-1)

                    elif layer == 2:
                        normalized_betas[layer][0][1:] = F.softmax (self.betas[layer][0][1:], dim=-1) * (2/3)
                        normalized_betas[layer][1] = F.softmax (self.betas[layer][1], dim=-1)
                        normalized_betas[layer][2] = F.softmax (self.betas[layer][2], dim=-1)
                    else :
                        normalized_betas[layer][0][1:] = F.softmax (self.betas[layer][0][1:], dim=-1) * (2/3)
                        normalized_betas[layer][1] = F.softmax (self.betas[layer][1], dim=-1)
                        normalized_betas[layer][2] = F.softmax (self.betas[layer][2], dim=-1)
                        normalized_betas[layer][3][:2] = F.softmax (self.betas[layer][3][:2], dim=-1) * (2/3)
        else:
            normalized_alphas_1 = alphas_1
            normalized_alphas_2 = alphas_2
            normalized_betas = betas

        for layer in range (self._num_layers) :

            if layer == 0 :
                level4_new, = self.cells[count] (temp, None, level_4[-1], None, normalized_alphas_1)
                count += 1
                level8_new, = self.cells[count] (temp, level_4[-1], None, None, normalized_alphas_1)
                count += 1

                level4_new = normalized_betas[layer][0][1] * level4_new
                level8_new = normalized_betas[layer][0][2] * level8_new

                
                level_4.append (level4_new)
                level_8.append (level8_new)

                del temp
                level_4_dense.append(self.dense_preprocess[layer][0](level4_new))
                level_8_dense.append(self.dense_preprocess[layer][1](level8_new))
                level_16_dense.append(self.dense_preprocess[layer][2](level8_new))
                level_32_dense.append(self.dense_preprocess[layer][3](level8_new))

            elif layer == 1 :
                level4_new_1, level4_new_2 = self.cells[count] (level_4[-2],
                                                                None,
                                                                level_4[-1],
                                                                level_8[-1],
                                                                normalized_alphas_1)
                level4_new = normalized_betas[layer][0][1] * level4_new_1 + normalized_betas[layer][1][0] * level4_new_2
                count += 1


                level8_new_1, level8_new_2 = self.cells[count] (level_4[-2],
                                                                level_4[-1],
                                                                level_8[-1],
                                                                None,
                                                                normalized_alphas_1)
                level8_new = normalized_betas[layer][0][2] * level8_new_1 + normalized_betas[layer][1][1] * level8_new_2
                count += 1


                level16_new, = self.cells[count] (level_4[-2],
                                                  level_8[-1],
                                                  None,
                                                  None,
                                                  normalized_alphas_1)

                level16_new = normalized_betas[layer][1][2] * level16_new
                count += 1


                level_4.append (level4_new)
                level_8.append (level8_new)
                level_16.append (level16_new)

                level_4_dense.append(self.dense_preprocess[layer][0](level4_new))
                level_8_dense.append(self.dense_preprocess[layer][1](level8_new))
                level_16_dense.append(self.dense_preprocess[layer][2](level16_new))
                level_32_dense.append(self.dense_preprocess[layer][3](level16_new))

            elif layer == 2 :
                level4_new_1, level4_new_2 = self.cells[count] (level_4[-2],
                                                                None,
                                                                level_4[-1],
                                                                level_8[-1],
                                                                normalized_alphas_1)
                count += 1
                level4_new = normalized_betas[layer][0][1] * level4_new_1 + normalized_betas[layer][1][0] * level4_new_2

                level8_new_1, level8_new_2, level8_new_3 = self.cells[count] (level_8[-2],
                                                                              level_4[-1],
                                                                              level_8[-1],
                                                                              level_16[-1],
                                                                              normalized_alphas_1)

                level8_new = normalized_betas[layer][0][2] * level8_new_1 + normalized_betas[layer][1][1] * level8_new_2 + normalized_betas[layer][2][0] * level8_new_3
                count += 1


                level16_new_1, level16_new_2 = self.cells[count] (level_8[-2],
                                                                  level_8[-1],
                                                                  level_16[-1],
                                                                  None,
                                                                  normalized_alphas_1)
                level16_new = normalized_betas[layer][1][2] * level16_new_1 + normalized_betas[layer][2][1] * level16_new_2
                count += 1


                level32_new, = self.cells[count] (level_8[-2],
                                                  level_16[-1],
                                                  None,
                                                  None,
                                                  normalized_alphas_1)

                level32_new = normalized_betas[layer][2][2] * level32_new
                count += 1


                level_4.append (level4_new)
                level_8.append (level8_new)
                level_16.append (level16_new)
                level_32.append (level32_new)

                level_4_dense.append(self.dense_preprocess[layer][0](level4_new))
                level_8_dense.append(self.dense_preprocess[layer][1](level8_new))
                level_16_dense.append(self.dense_preprocess[layer][2](level16_new))
                level_32_dense.append(self.dense_preprocess[layer][3](level32_new))

            elif layer == 3 :
                level4_new_1, level4_new_2 = self.cells[count] (torch.cat(level_4_dense[:-1], dim=1),
                                                                None,
                                                                level_4[-1],
                                                                level_8[-1],
                                                                normalized_alphas_1)

                level4_new = normalized_betas[layer][0][1] * level4_new_1 + normalized_betas[layer][1][0] * level4_new_2
                count += 1


                level8_new_1, level8_new_2, level8_new_3 = self.cells[count] (torch.cat(level_8_dense[:-1], dim=1),
                                                                              level_4[-1],
                                                                              level_8[-1],
                                                                              level_16[-1],
                                                                              normalized_alphas_1)

                level8_new = normalized_betas[layer][0][2] * level8_new_1 + normalized_betas[layer][1][1] * level8_new_2 + normalized_betas[layer][2][0] * level8_new_3
                count += 1

                level16_new_1, level16_new_2, level16_new_3 = self.cells[count] (torch.cat(level_16_dense[:-1], dim=1),
                                                                                 level_8[-1],
                                                                                 level_16[-1],
                                                                                 level_32[-1],
                                                                                 normalized_alphas_1)

                level16_new = normalized_betas[layer][1][2] * level16_new_1 + normalized_betas[layer][2][1] * level16_new_2 + normalized_betas[layer][3][0] * level16_new_3
                count += 1


                level32_new_1, level32_new_2 = self.cells[count] (torch.cat(level_32_dense[:-1], dim=1),
                                                                  level_16[-1],
                                                                  level_32[-1],
                                                                  None,
                                                                  normalized_alphas_1)

                level32_new = normalized_betas[layer][2][2] * level32_new_1 + normalized_betas[layer][3][1] * level32_new_2
                count += 1


                level_4.append (level4_new)
                level_8.append (level8_new)
                level_16.append (level16_new)
                level_32.append (level32_new)

                level_4_dense.append(self.dense_preprocess[layer][0](level4_new))
                level_8_dense.append(self.dense_preprocess[layer][1](level8_new))
                level_16_dense.append(self.dense_preprocess[layer][2](level16_new))
                level_32_dense.append(self.dense_preprocess[layer][3](level32_new))

            elif layer < self.exit_layer :
                level4_new_1, level4_new_2 = self.cells[count] (torch.cat(level_4_dense[:-1], dim=1),
                                                                None,
                                                                level_4[-1],
                                                                level_8[-1],
                                                                normalized_alphas_1)

                level4_new = normalized_betas[layer][0][1] * level4_new_1 + normalized_betas[layer][1][0] * level4_new_2
                count += 1


                level8_new_1, level8_new_2, level8_new_3 = self.cells[count] (torch.cat(level_8_dense[:-1], dim=1),
                                                                              level_4[-1],
                                                                              level_8[-1],
                                                                              level_16[-1],
                                                                              normalized_alphas_1)

                level8_new = normalized_betas[layer][0][2] * level8_new_1 + normalized_betas[layer][1][1] * level8_new_2 + normalized_betas[layer][2][0] * level8_new_3
                count += 1

                level16_new_1, level16_new_2, level16_new_3 = self.cells[count] (torch.cat(level_16_dense[:-1], dim=1),
                                                                                 level_8[-1],
                                                                                 level_16[-1],
                                                                                 level_32[-1],
                                                                                 normalized_alphas_1)

                level16_new = normalized_betas[layer][1][2] * level16_new_1 + normalized_betas[layer][2][1] * level16_new_2 + normalized_betas[layer][3][0] * level16_new_3
                count += 1


                level32_new_1, level32_new_2 = self.cells[count] (torch.cat(level_32_dense[:-1], dim=1),
                                                                  level_16[-1],
                                                                  level_32[-1],
                                                                  None,
                                                                  normalized_alphas_1)

                level32_new = normalized_betas[layer][2][2] * level32_new_1 + normalized_betas[layer][3][1] * level32_new_2
                count += 1

                level_4.append (level4_new)
                level_8.append (level8_new)
                level_16.append (level16_new)
                level_32.append (level32_new)

                level_4_dense.append(self.dense_preprocess[layer][0](level4_new))
                level_8_dense.append(self.dense_preprocess[layer][1](level8_new))
                level_16_dense.append(self.dense_preprocess[layer][2](level16_new))
                level_32_dense.append(self.dense_preprocess[layer][3](level32_new))

            elif layer == self.exit_layer:
                level4_new_1, level4_new_2 = self.cells[count] (torch.cat(level_4_dense[:-1], dim=1),
                                                                None,
                                                                level_4[-1],
                                                                level_8[-1],
                                                                normalized_alphas_1)

                level4_new = normalized_betas[layer][0][1] * level4_new_1 + normalized_betas[layer][1][0] * level4_new_2
                count += 1


                level8_new_1, level8_new_2, level8_new_3 = self.cells[count] (torch.cat(level_8_dense[:-1], dim=1),
                                                                              level_4[-1],
                                                                              level_8[-1],
                                                                              level_16[-1],
                                                                              normalized_alphas_1)

                level8_new = normalized_betas[layer][0][2] * level8_new_1 + normalized_betas[layer][1][1] * level8_new_2 + normalized_betas[layer][2][0] * level8_new_3
                count += 1


                level16_new_1, level16_new_2, level16_new_3 = self.cells[count] (torch.cat(level_16_dense[:-1], dim=1),
                                                                                 level_8[-1],
                                                                                 level_16[-1],
                                                                                 level_32[-1],
                                                                                 normalized_alphas_1)

                level16_new = normalized_betas[layer][1][2] * level16_new_1 + normalized_betas[layer][2][1] * level16_new_2 + normalized_betas[layer][3][0] * level16_new_3
                count += 1


                level32_new_1, level32_new_2 = self.cells[count] (torch.cat(level_32_dense[:-1], dim=1),
                                                                  level_16[-1],
                                                                  level_32[-1],
                                                                  None,
                                                                  normalized_alphas_1)

                level32_new = normalized_betas[layer][2][2] * level32_new_1 + normalized_betas[layer][3][1] * level32_new_2
                count += 1


                level_4.append (level4_new)
                level_8.append (level8_new)
                level_16.append (level16_new)
                level_32.append (level32_new)


                level_4_dense.append(self.dense_preprocess[layer][0](level4_new))
                level_8_dense.append(self.dense_preprocess[layer][1](level8_new))
                level_16_dense.append(self.dense_preprocess[layer][2](level16_new))
                level_32_dense.append(self.dense_preprocess[layer][3](level32_new))

                # exit_1_4_new = self.aspp_exit_1_4(level_4[-1])
                exit_1_8_new = self.aspp_exit_1_8(level_8[-1])
                # exit_1_16_new = self.aspp_exit_1_16(level_16[-1])
                # exit_1_32_new = self.aspp_exit_1_32(level_32[-1])

            elif layer == self.exit_layer+1:
                level4_new_1, level4_new_2 = self.cells[count] (torch.cat(level_4_dense[:-1], dim=1),
                                                                None,
                                                                level_4[-1],
                                                                level_8[-1],
                                                                normalized_alphas_2)
                level4_new = normalized_betas[layer][0][1] * level4_new_1 + normalized_betas[layer][1][0] * level4_new_2
                count += 1


                level8_new_1, level8_new_2, level8_new_3 = self.cells[count] (torch.cat(level_8_dense[:-1], dim=1),
                                                                              level_4[-1],
                                                                              level_8[-1],
                                                                              level_16[-1],
                                                                              normalized_alphas_2)
                level8_new = normalized_betas[layer][0][2] * level8_new_1 + normalized_betas[layer][1][1] * level8_new_2 + normalized_betas[layer][2][0] * level8_new_3
                count += 1


                level16_new_1, level16_new_2, level16_new_3 = self.cells[count] (torch.cat(level_16_dense[:-1], dim=1),
                                                                                 level_8[-1],
                                                                                 level_16[-1],
                                                                                 level_32[-1],
                                                                                 normalized_alphas_2)
                level16_new = normalized_betas[layer][1][2] * level16_new_1 + normalized_betas[layer][2][1] * level16_new_2 + normalized_betas[layer][3][0] * level16_new_3
                count += 1


                level32_new_1, level32_new_2 = self.cells[count] (torch.cat(level_32_dense[:-1], dim=1),
                                                                  level_16[-1],
                                                                  level_32[-1],
                                                                  None,
                                                                  normalized_alphas_2)
                level32_new = normalized_betas[layer][2][2] * level32_new_1 + normalized_betas[layer][3][1] * level32_new_2
                count += 1


                level_4.append (level4_new)
                level_8.append (level8_new)
                level_16.append (level16_new)
                level_32.append (level32_new)

                level_4_dense.append(self.dense_preprocess[layer][0](level4_new))
                level_8_dense.append(self.dense_preprocess[layer][1](level8_new))
                level_16_dense.append(self.dense_preprocess[layer][2](level16_new))
                level_32_dense.append(self.dense_preprocess[layer][3](level32_new))

            elif layer == self.exit_layer+2:
                level4_new_1, level4_new_2 = self.cells[count] (torch.cat(level_4_dense[:-1], dim=1),
                                                                None,
                                                                level_4[-1],
                                                                level_8[-1],
                                                                normalized_alphas_2)
                level4_new = normalized_betas[layer][0][1] * level4_new_1 + normalized_betas[layer][1][0] * level4_new_2
                count += 1


                level8_new_1, level8_new_2, level8_new_3 = self.cells[count] (torch.cat(level_8_dense[:-1], dim=1),
                                                                              level_4[-1],
                                                                              level_8[-1],
                                                                              level_16[-1],
                                                                              normalized_alphas_2)
                level8_new = normalized_betas[layer][0][2] * level8_new_1 + normalized_betas[layer][1][1] * level8_new_2 + normalized_betas[layer][2][0] * level8_new_3
                count += 1


                level16_new_1, level16_new_2, level16_new_3 = self.cells[count] (torch.cat(level_16_dense[:-1], dim=1),
                                                                                 level_8[-1],
                                                                                 level_16[-1],
                                                                                 level_32[-1],
                                                                                 normalized_alphas_2)
                level16_new = normalized_betas[layer][1][2] * level16_new_1 + normalized_betas[layer][2][1] * level16_new_2 + normalized_betas[layer][3][0] * level16_new_3
                count += 1


                level32_new_1, level32_new_2 = self.cells[count] (torch.cat(level_32_dense[:-1], dim=1),
                                                                  level_16[-1],
                                                                  level_32[-1],
                                                                  None,
                                                                  normalized_alphas_2)
                level32_new = normalized_betas[layer][2][2] * level32_new_1 + normalized_betas[layer][3][1] * level32_new_2
                count += 1


                level_4.append (level4_new)
                level_8.append (level8_new)
                level_16.append (level16_new)
                level_32.append (level32_new)

                level_4_dense.append(self.dense_preprocess[layer][0](level4_new))
                level_8_dense.append(self.dense_preprocess[layer][1](level8_new))
                level_16_dense.append(self.dense_preprocess[layer][2](level16_new))
                level_32_dense.append(self.dense_preprocess[layer][3](level32_new))

            elif layer == self._num_layers-1:
                level4_new_1, level4_new_2 = self.cells[count] (torch.cat(level_4_dense, dim=1),
                                                                None,
                                                                level_4[-1],
                                                                level_8[-1],
                                                                normalized_alphas_2)
                level4_new = normalized_betas[layer][0][1] * level4_new_1 + normalized_betas[layer][1][0] * level4_new_2
                count += 1


                level8_new_1, level8_new_2, level8_new_3 = self.cells[count] (torch.cat(level_8_dense, dim=1),
                                                                              level_4[-1],
                                                                              level_8[-1],
                                                                              level_16[-1],
                                                                              normalized_alphas_2)
                level8_new = normalized_betas[layer][0][2] * level8_new_1 + normalized_betas[layer][1][1] * level8_new_2 + normalized_betas[layer][2][0] * level8_new_3
                count += 1


                level16_new_1, level16_new_2, level16_new_3 = self.cells[count] (torch.cat(level_16_dense, dim=1),
                                                                                 level_8[-1],
                                                                                 level_16[-1],
                                                                                 level_32[-1],
                                                                                 normalized_alphas_2)
                level16_new = normalized_betas[layer][1][2] * level16_new_1 + normalized_betas[layer][2][1] * level16_new_2 + normalized_betas[layer][3][0] * level16_new_3
                count += 1


                level32_new_1, level32_new_2 = self.cells[count] (torch.cat(level_32_dense, dim=1),
                                                                  level_16[-1],
                                                                  level_32[-1],
                                                                  None,
                                                                  normalized_alphas_2)
                level32_new = normalized_betas[layer][2][2] * level32_new_1 + normalized_betas[layer][3][1] * level32_new_2
                count += 1


                level_4.append (level4_new)
                level_8.append (level8_new)
                level_16.append (level16_new)
                level_32.append (level32_new)

            else :
                level4_new_1, level4_new_2 = self.cells[count] (torch.cat(level_4_dense[:-1], dim=1),
                                                                None,
                                                                level_4[-1],
                                                                level_8[-1],
                                                                normalized_alphas_2)
                level4_new = normalized_betas[layer][0][1] * level4_new_1 + normalized_betas[layer][1][0] * level4_new_2
                count += 1


                level8_new_1, level8_new_2, level8_new_3 = self.cells[count] (torch.cat(level_8_dense[:-1], dim=1),
                                                                              level_4[-1],
                                                                              level_8[-1],
                                                                              level_16[-1],
                                                                              normalized_alphas_2)
                level8_new = normalized_betas[layer][0][2] * level8_new_1 + normalized_betas[layer][1][1] * level8_new_2 + normalized_betas[layer][2][0] * level8_new_3
                count += 1


                level16_new_1, level16_new_2, level16_new_3 = self.cells[count] (torch.cat(level_16_dense[:-1], dim=1),
                                                                                 level_8[-1],
                                                                                 level_16[-1],
                                                                                 level_32[-1],
                                                                                 normalized_alphas_2)
                level16_new = normalized_betas[layer][1][2] * level16_new_1 + normalized_betas[layer][2][1] * level16_new_2 + normalized_betas[layer][3][0] * level16_new_3
                count += 1


                level32_new_1, level32_new_2 = self.cells[count] (torch.cat(level_32_dense[:-1], dim=1),
                                                                  level_16[-1],
                                                                  level_32[-1],
                                                                  None,
                                                                  normalized_alphas_2)
                level32_new = normalized_betas[layer][2][2] * level32_new_1 + normalized_betas[layer][3][1] * level32_new_2
                count += 1


                level_4.append (level4_new)
                level_8.append (level8_new)
                level_16.append (level16_new)
                level_32.append (level32_new)

                if layer < self._num_layers -2:
                    level_4_dense.append(self.dense_preprocess[layer][0](level4_new))
                    level_8_dense.append(self.dense_preprocess[layer][1](level8_new))
                    level_16_dense.append(self.dense_preprocess[layer][2](level16_new))
                    level_32_dense.append(self.dense_preprocess[layer][3](level32_new))

            if layer < self.exit_layer:
                level_4 = level_4[-2:]
                level_8 = level_8[-2:]
                level_16 = level_16[-2:]
                evel_32 = level_32[-2:]
            else:
                level_4 = level_4[-1:]
                level_8 = level_8[-1:]
                evel_16 = level_16[-1:]
                level_32 = level_32[-1:]

        # exit_2_4_new = self.aspp_exit_2_4 (level_4[-1])
        del level_4
        exit_2_8_new = self.aspp_exit_2_8 (level_8[-1])
        del level_8
        # exit_2_16_new = self.aspp_exit_2_16 (level_16[-1])
        del level_16
        # exit_2_32_new = self.aspp_exit_2_32 (level_32[-1])
        del level_32

        upsample = nn.Upsample(size=x.size()[2:], mode='bilinear', align_corners=True)
        # exit_2_4_new = upsample (exit_2_4_new)
        exit_2_8_new = upsample (exit_2_8_new)
        # exit_2_16_new = upsample (exit_2_16_new)
        # exit_2_32_new = upsample (exit_2_32_new)

        # exit_1_4_new = upsample(exit_1_4_new)
        exit_1_8_new = upsample(exit_1_8_new)
        # exit_1_16_new = upsample(exit_1_16_new)
        # exit_1_32_new = upsample(exit_1_32_new)

        # exit_1_sum_feature_map = exit_1_4_new + exit_1_8_new + exit_1_16_new + exit_1_32_new
        # del exit_1_4_new, exit_1_8_new, exit_1_16_new, exit_1_32_new

        # exit_2_sum_feature_map = exit_2_4_new + exit_2_8_new + exit_2_16_new + exit_2_32_new


        return exit_1_8_new, exit_2_8_new


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine != False:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()


    def _initialize_alphas_betas(self):
        k_1 = sum(1 for i in range(self.B_1) for n in range(2+i))
        k_2 = sum(1 for i in range(self.B_2) for n in range(2+i))
        num_ops = len(PRIMITIVES)
        alphas_1 = torch.tensor (1e-3*torch.randn(k_1, num_ops).cuda(), requires_grad=True)
        alphas_2 = torch.tensor (1e-3*torch.randn(k_2, num_ops).cuda(), requires_grad=True)
        betas = torch.tensor (1e-3*torch.randn(12, 4, 3).cuda(), requires_grad=True)

        self._arch_parameters = [
            alphas_1,
            alphas_2,
            betas,
            ]
        self._arch_param_names = [
            'alphas_1',
            'alphas_2',
            'betas',
            ]

        [self.register_parameter(name, torch.nn.Parameter(param)) for name, param in zip(self._arch_param_names, self._arch_parameters)]


    def arch_parameters (self) :
        return [param for name, param in self.named_parameters() if name in self._arch_param_names]


    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if name not in self._arch_param_names]

    def is_train(self, train=True):
        if train:
            for cell in self.cells:
                cell.train_mode = True
        else:
            for cell in self.cells:
                cell.train_mode = False


def main () :
    model = Model_search (7, 12, None)
    x = torch.tensor (torch.ones (4, 3, 224, 224))

if __name__ == '__main__' :
    main ()