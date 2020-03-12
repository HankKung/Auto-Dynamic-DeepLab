import torch
import torch.nn as nn
import numpy as np
from modeling.genotypes import PRIMITIVES
import torch.nn.functional as F
from modeling.operations import *
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class Cell_fixed_baseline(nn.Module):

    def __init__(self,
                B, 
                prev_prev_C,
                prev_C_down,
                prev_C_same,
                prev_C_up, 
                C_out,
                cell,
                BatchNorm=nn.BatchNorm2d,
                pre_preprocess_sample_rate=1):

        super(Cell_fixed_baseline, self).__init__()
        eps = 1e-5
        momentum = 0.1

        self.B = B
        self.cell_arch = cell
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

    def prev_feature_resize(self, prev_feature, mode):
        if mode == 'down':
            feature_size_h = self.scale_dimension(prev_feature.shape[2], 0.5)
            feature_size_w = self.scale_dimension(prev_feature.shape[3], 0.5)
        elif mode == 'up':
            feature_size_h = self.scale_dimension(prev_feature.shape[2], 2)
            feature_size_w = self.scale_dimension(prev_feature.shape[3], 2)
        return F.interpolate(prev_feature, (feature_size_h, feature_size_w), mode='bilinear')


    def forward(self, s0, s1_down, s1_same, s1_up):

        if s1_down is not None:
            s1_down = self.preprocess_down(s1_down)
            size_h, size_w = s1_down.shape[2], s1_down.shape[3]
        if s1_same is not None:
            s1_same = self.preprocess_same(s1_same)
            size_h, size_w = s1_same.shape[2], s1_same.shape[3]
        if s1_up is not None:
            s1_up = self.prev_feature_resize(s1_up, 'up')
            s1_up = self.preprocess_up(s1_up)
            size_h, size_w = s1_up.shape[2], s1_up.shape[3]

        all_states = []
        if s0 is not None:
            s0 = F.interpolate(s0, (size_h, size_w), mode='bilinear') if (
                s0.shape[2] < size_h) or (s0.shape[3] < size_w) else s0
            s0 = self.pre_preprocess(s0)
            if s1_down is not None:
                states_down = [s0, s1_down]
                all_states.append(states_down)
                del s1_down
            if s1_same is not None:
                states_same = [s0, s1_same]
                all_states.append(states_same)
                del s1_same
            if s1_up is not None:
                states_up = [s0, s1_up]
                all_states.append(states_up)
                del s1_up
        else:
            if s1_down is not None:
                states_down = [0, s1_down]
                all_states.append(states_down)
            if s1_same is not None:
                states_same = [0, s1_same]
                all_states.append(states_same)
            if s1_up is not None:
                states_up = [0, s1_up]
                all_states.append(states_up)
        del s0
        final_concates = []

        for states in all_states:
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
            final_concates.append(concat_feature)
        return final_concates


class Model_layer_search_baseline (nn.Module) :
    def __init__(self,
                num_classes,
                num_layers,
                args,
                exit_layer=5,
                alphas=None):

        super(Model_layer_search_baseline, self).__init__()
        cell = Cell_fixed_baseline
        BatchNorm = SynchronizedBatchNorm2d if args.sync_bn == True else nn.BatchNorm2d
        self.cells = nn.ModuleList()
        self._num_layers = num_layers
        self._num_classes = num_classes
        self.exit_layer = exit_layer
        self._initialize_alphas_betas()
        self.alphas = alphas
        B = args.B
        F = args.F
        f_initial = F * B
        half_f_initial = int(f_initial / 2)

        FB = F * B

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, half_f_initial, 3, stride=2, padding=1, bias=False),
            BatchNorm(half_f_initial),
        )
        self.stem1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(half_f_initial, f_initial, 3, stride=2, padding=1, bias=False),
            BatchNorm(f_initial),
        )

        """ build the cells """
        for i in range (self._num_layers):

            if i == 0 :
                cell1 = cell (B, half_f_initial,
                              None, f_initial, None,
                              F, alphas, BatchNorm=BatchNorm, pre_preprocess_sample_rate=0.5)
                cell2 = cell (B, half_f_initial,
                              f_initial, None, None,
                              F * 2, alphas, BatchNorm=BatchNorm, pre_preprocess_sample_rate=0.25)
                self.cells += [cell1]
                self.cells += [cell2]
            elif i == 1 :
                cell1 = cell (B, f_initial,
                              None, FB, FB * 2,
                              F, alphas, BatchNorm=BatchNorm)

                cell2 = cell (B, f_initial,
                              FB, FB * 2, None,
                              F * 2, alphas, BatchNorm=BatchNorm, pre_preprocess_sample_rate=0.5)

                cell3 = cell (B, f_initial,
                              FB * 2, None, None,
                              F * 4, alphas, BatchNorm=BatchNorm, pre_preprocess_sample_rate=0.25)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]

            elif i == 2 :
                cell1 = cell (B, FB,
                              None, FB, FB * 2,
                              F, alphas, BatchNorm=BatchNorm)

                cell2 = cell (B, FB * 2,
                              FB, FB * 2, FB * 4,
                              F * 2, alphas, BatchNorm=BatchNorm)

                cell3 = cell (B, FB * 2,
                              FB * 2, FB * 4, None,
                              F * 4, alphas, BatchNorm=BatchNorm, pre_preprocess_sample_rate=0.5)

                cell4 = cell (B, FB * 2,
                              FB * 4, None, None,
                              F * 8, alphas, BatchNorm=BatchNorm, pre_preprocess_sample_rate=0.25)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]

            elif i == 3 :
                cell1 = cell (B, FB,
                              None, FB, FB * 2,
                              F, alphas, BatchNorm=BatchNorm)

                cell2 = cell (B, FB * 2,
                              FB, FB * 2, FB * 4,
                              F * 2, alphas, BatchNorm=BatchNorm)

                cell3 = cell (B, FB * 4,
                              FB * 2, FB * 4, None,
                              F * 4, alphas, BatchNorm=BatchNorm, pre_preprocess_sample_rate=0.5)

                cell4 = cell (B, FB * 4,
                              FB * 4, None, None,
                              F * 8, alphas, BatchNorm=BatchNorm, pre_preprocess_sample_rate=0.25)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]

            else:
                cell1 = cell (B, F * B,
                              None, FB, FB * 2,
                              F, alphas, BatchNorm=BatchNorm)

                cell2 = cell (B, F * B * 2,
                              FB, FB * 2, FB * 4,
                              F * 2, alphas, BatchNorm=BatchNorm)

                cell3 = cell (B, F * B * 4,
                              FB * 2, FB * 4, FB * 8,
                              F * 4, alphas, BatchNorm=BatchNorm)

                cell4 = cell (B, F * B * 8,
                              FB * 4, FB * 8, None,
                              F * 8, alphas, BatchNorm=BatchNorm)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]

        self.aspp_4 = ASPP (FB, self._num_classes, 24, 24, BatchNorm=BatchNorm) #96 / 4 as in the paper
        self.aspp_8 = ASPP (FB * 2, self._num_classes, 12, 12, BatchNorm=BatchNorm) #96 / 8
        self.aspp_16 = ASPP (FB * 4, self._num_classes, 6, 6, BatchNorm=BatchNorm) #96 / 16
        self.aspp_32 = ASPP (FB * 8, self._num_classes, 3, 3, BatchNorm=BatchNorm) #96 / 32
        self._init_weight()


    def forward (self, x) :
        level_4 = []
        level_8 = []
        level_16 = []
        level_32 = []

        temp = self.stem0 (x)
        level_4.append (self.stem1(temp))

        count = 0

        normalized_betas = torch.randn(12, 4, 3).cuda().half()

        """ Softmax on betas """
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


        for layer in range (self._num_layers) :

            if layer == 0 :
                level4_new, = self.cells[count] (temp, None, level_4[-1], None)
                count += 1
                level8_new, = self.cells[count] (temp, level_4[-1], None, None)
                count += 1

                level4_new = normalized_betas[layer][0][1] * level4_new
                level8_new = normalized_betas[layer][0][2] * level8_new

                
                level_4.append (level4_new)
                level_8.append (level8_new)

                del temp

            elif layer == 1 :
                level4_new_1, level4_new_2 = self.cells[count] (level_4[-2],
                                                                None,
                                                                level_4[-1],
                                                                level_8[-1])
                level4_new = normalized_betas[layer][0][1] * level4_new_1 + normalized_betas[layer][1][0] * level4_new_2
                count += 1


                level8_new_1, level8_new_2 = self.cells[count] (level_4[-2],
                                                                level_4[-1],
                                                                level_8[-1],
                                                                None)
                level8_new = normalized_betas[layer][0][2] * level8_new_1 + normalized_betas[layer][1][1] * level8_new_2
                count += 1


                level16_new, = self.cells[count] (level_4[-2],
                                                  level_8[-1],
                                                  None,
                                                  None)

                level16_new = normalized_betas[layer][1][2] * level16_new
                count += 1


                level_4.append (level4_new)
                level_8.append (level8_new)
                level_16.append (level16_new)


            elif layer == 2 :
                level4_new_1, level4_new_2 = self.cells[count] (level_4[-2],
                                                                None,
                                                                level_4[-1],
                                                                level_8[-1])
                count += 1
                level4_new = normalized_betas[layer][0][1] * level4_new_1 + normalized_betas[layer][1][0] * level4_new_2

                level8_new_1, level8_new_2, level8_new_3 = self.cells[count] (level_8[-2],
                                                                              level_4[-1],
                                                                              level_8[-1],
                                                                              level_16[-1])

                level8_new = normalized_betas[layer][0][2] * level8_new_1 + normalized_betas[layer][1][1] * level8_new_2 + normalized_betas[layer][2][0] * level8_new_3
                count += 1


                level16_new_1, level16_new_2 = self.cells[count] (level_8[-2],
                                                                  level_8[-1],
                                                                  level_16[-1],
                                                                  None)
                level16_new = normalized_betas[layer][1][2] * level16_new_1 + normalized_betas[layer][2][1] * level16_new_2
                count += 1


                level32_new, = self.cells[count] (level_8[-2],
                                                  level_16[-1],
                                                  None,
                                                  None)

                level32_new = normalized_betas[layer][2][2] * level32_new
                count += 1


                level_4.append (level4_new)
                level_8.append (level8_new)
                level_16.append (level16_new)
                level_32.append (level32_new)


            elif layer == 3 :
                level4_new_1, level4_new_2 = self.cells[count] (level_4[-2],
                                                                None,
                                                                level_4[-1],
                                                                level_8[-1])

                level4_new = normalized_betas[layer][0][1] * level4_new_1 + normalized_betas[layer][1][0] * level4_new_2
                count += 1


                level8_new_1, level8_new_2, level8_new_3 = self.cells[count] (level_8[-2],
                                                                              level_4[-1],
                                                                              level_8[-1],
                                                                              level_16[-1])

                level8_new = normalized_betas[layer][0][2] * level8_new_1 + normalized_betas[layer][1][1] * level8_new_2 + normalized_betas[layer][2][0] * level8_new_3
                count += 1

                level16_new_1, level16_new_2, level16_new_3 = self.cells[count] (level_16[-2],
                                                                                 level_8[-1],
                                                                                 level_16[-1],
                                                                                 level_32[-1])

                level16_new = normalized_betas[layer][1][2] * level16_new_1 + normalized_betas[layer][2][1] * level16_new_2 + normalized_betas[layer][3][0] * level16_new_3
                count += 1


                level32_new_1, level32_new_2 = self.cells[count] (level_16[-2],
                                                                  level_16[-1],
                                                                  level_32[-1],
                                                                  None)

                level32_new = normalized_betas[layer][2][2] * level32_new_1 + normalized_betas[layer][3][1] * level32_new_2
                count += 1


                level_4.append (level4_new)
                level_8.append (level8_new)
                level_16.append (level16_new)
                level_32.append (level32_new)


            elif layer < self.exit_layer :
                level4_new_1, level4_new_2 = self.cells[count] (level_4[-2],
                                                                None,
                                                                level_4[-1],
                                                                level_8[-1])

                level4_new = normalized_betas[layer][0][1] * level4_new_1 + normalized_betas[layer][1][0] * level4_new_2
                count += 1


                level8_new_1, level8_new_2, level8_new_3 = self.cells[count] (level_8[-2],
                                                                              level_4[-1],
                                                                              level_8[-1],
                                                                              level_16[-1])

                level8_new = normalized_betas[layer][0][2] * level8_new_1 + normalized_betas[layer][1][1] * level8_new_2 + normalized_betas[layer][2][0] * level8_new_3
                count += 1

                level16_new_1, level16_new_2, level16_new_3 = self.cells[count] (level_16[-2],
                                                                                 level_8[-1],
                                                                                 level_16[-1],
                                                                                 level_32[-1])

                level16_new = normalized_betas[layer][1][2] * level16_new_1 + normalized_betas[layer][2][1] * level16_new_2 + normalized_betas[layer][3][0] * level16_new_3
                count += 1


                level32_new_1, level32_new_2 = self.cells[count] (torch.cat(level_32[-2]),
                                                                  level_16[-1],
                                                                  level_32[-1],
                                                                  None)

                level32_new = normalized_betas[layer][2][2] * level32_new_1 + normalized_betas[layer][3][1] * level32_new_2
                count += 1

                level_4.append (level4_new)
                level_8.append (level8_new)
                level_16.append (level16_new)
                level_32.append (level32_new)


            elif layer == self.exit_layer:
                level4_new_1, level4_new_2 = self.cells[count] (level_4[-2],
                                                                None,
                                                                level_4[-1],
                                                                level_8[-1])

                level4_new = normalized_betas[layer][0][1] * level4_new_1 + normalized_betas[layer][1][0] * level4_new_2
                count += 1


                level8_new_1, level8_new_2, level8_new_3 = self.cells[count] (level_8[:-1],
                                                                              level_4[-1],
                                                                              level_8[-1],
                                                                              level_16[-1])

                level8_new = normalized_betas[layer][0][2] * level8_new_1 + normalized_betas[layer][1][1] * level8_new_2 + normalized_betas[layer][2][0] * level8_new_3
                count += 1


                level16_new_1, level16_new_2, level16_new_3 = self.cells[count] (level_16[-2],
                                                                                 level_8[-1],
                                                                                 level_16[-1],
                                                                                 level_32[-1])

                level16_new = normalized_betas[layer][1][2] * level16_new_1 + normalized_betas[layer][2][1] * level16_new_2 + normalized_betas[layer][3][0] * level16_new_3
                count += 1


                level32_new_1, level32_new_2 = self.cells[count] (level_32[-2],
                                                                  level_16[-1],
                                                                  level_32[-1],
                                                                  None)

                level32_new = normalized_betas[layer][2][2] * level32_new_1 + normalized_betas[layer][3][1] * level32_new_2
                count += 1


                level_4.append (level4_new)
                level_8.append (level8_new)
                level_16.append (level16_new)
                level_32.append (level32_new)

                exit_1_4_new = self.aspp_4(level_4[-1])
                exit_1_8_new = self.aspp_8(level_8[-1])
                exit_1_16_new = self.aspp_16(level_16[-1])
                exit_1_32_new = self.aspp_32(level_32[-1])


            else :
                level4_new_1, level4_new_2 = self.cells[count] (torch.cat(level_4[-2], dim=1),
                                                                None,
                                                                level_4[-1],
                                                                level_8[-1])
                level4_new = normalized_betas[layer][0][1] * level4_new_1 + normalized_betas[layer][1][0] * level4_new_2
                count += 1


                level8_new_1, level8_new_2, level8_new_3 = self.cells[count] (torch.cat(level_8[-2], dim=1),
                                                                              level_4[-1],
                                                                              level_8[-1],
                                                                              level_16[-1])
                level8_new = normalized_betas[layer][0][2] * level8_new_1 + normalized_betas[layer][1][1] * level8_new_2 + normalized_betas[layer][2][0] * level8_new_3
                count += 1


                level16_new_1, level16_new_2, level16_new_3 = self.cells[count] (torch.cat(level_16[-2], dim=1),
                                                                                 level_8[-1],
                                                                                 level_16[-1],
                                                                                 level_32[-1])
                level16_new = normalized_betas[layer][1][2] * level16_new_1 + normalized_betas[layer][2][1] * level16_new_2 + normalized_betas[layer][3][0] * level16_new_3
                count += 1


                level32_new_1, level32_new_2 = self.cells[count] (torch.cat(level_32[-2], dim=1),
                                                                  level_16[-1],
                                                                  level_32[-1],
                                                                  None)
                level32_new = normalized_betas[layer][2][2] * level32_new_1 + normalized_betas[layer][3][1] * level32_new_2
                count += 1


                level_4.append (level4_new)
                level_8.append (level8_new)
                level_16.append (level16_new)
                level_32.append (level32_new)


            level_4 = level_4[-2:]
            level_8 = level_8[-2:]
            level_16 = level_16[-2:]
            level_32 = level_32[-2:]


        exit_2_4_new = self.aspp_4 (level_4[-1])
        exit_2_8_new = self.aspp_8 (level_8[-1])
        exit_2_16_new = self.aspp_16 (level_16[-1])
        exit_2_32_new = self.aspp_32 (level_32[-1])

        upsample = nn.Upsample(size=x.size()[2:], mode='bilinear', align_corners=True)
        exit_2_4_new = upsample (exit_2_4_new)
        exit_2_8_new = upsample (exit_2_8_new)
        exit_2_16_new = upsample (exit_2_16_new)
        exit_2_32_new = upsample (exit_2_32_new)

        exit_1_4_new = upsample(exit_1_4_new)
        exit_1_8_new = upsample(exit_1_8_new)
        exit_1_16_new = upsample(exit_1_16_new)
        exit_1_32_new = upsample(exit_1_32_new)

        exit_1_sum_feature_map = exit_1_4_new + exit_1_8_new + exit_1_16_new + exit_1_32_new

        exit_2_sum_feature_map = exit_2_4_new + exit_2_8_new + exit_2_16_new + exit_2_32_new


        return exit_1_sum_feature_map, exit_2_sum_feature_map


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
        betas = torch.tensor (1e-3*torch.randn(12, 4, 3).cuda(), requires_grad=True)
        self._arch_parameters = [betas]
        self._arch_param_names = ['betas']
        [self.register_parameter(name, torch.nn.Parameter(param)) for name, param in zip(self._arch_param_names, self._arch_parameters)]


    def arch_parameters (self) :
        return [param for name, param in self.named_parameters() if name in self._arch_param_names]


    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if name not in self._arch_param_names]


def main () :
    model = Model_search (7, 12, None)
    x = torch.tensor (torch.ones (4, 3, 224, 224))

if __name__ == '__main__' :
    main ()