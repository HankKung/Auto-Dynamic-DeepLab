import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modeling.operations import *
from modeling.genotypes import PRIMITIVES
from modeling.genotypes import Genotype


class MixedOp (nn.Module):

    def __init__(self, C, stride, BatchNorm):
        super(MixedOp, self).__init__()
        eps=1e-5
        momentum=0.1
        self._ops = nn.ModuleList()

        for i, primitive in enumerate(PRIMITIVES):
            op = OPS[primitive](C, stride, BatchNorm, eps, momentum, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, BatchNorm(C, eps=eps, momentum=momentum, affine=False))
            self._ops.append(op)

    def forward(self, x, weights, training=True):
        if training:
            return sum(w * op(x) for w, op in zip(weights, self._ops))
        else:
            w = torch.argmax(weights)
            return  self._ops[w](x)

            
class Cell(nn.Module):

    def __init__(self,
                B,
                prev_prev_C,
                prev_C_down,
                prev_C_same,
                prev_C_up,
                C_out,
                BatchNorm=nn.BatchNorm2d,
                pre_preprocess_sample_rate=1):

        super(Cell, self).__init__()

        if prev_C_down is not None:  
            self.preprocess_down = FactorizedReduce(
                prev_C_down, C_out, BatchNorm=BatchNorm, affine=False)
        if prev_C_same is not None:
            self.preprocess_same = ReLUConvBN(
                prev_C_same, C_out, 1, 1, 0, BatchNorm=BatchNorm, affine=False)
        if prev_C_up is not None:
            self.preprocess_up = ReLUConvBN(
                prev_C_up, C_out, 1, 1, 0, BatchNorm=BatchNorm, affine=False)

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

        self.B = B
        self._ops = nn.ModuleList()

        for i in range(self.B):
            for j in range(2+i):
                stride = 1
                if prev_prev_C == -1 and j == 0:
                    op = None
                else:
                    op = MixedOp(C_out, stride, BatchNorm)
                self._ops.append(op)


    def scale_dimension(self, dim, scale):
        assert isinstance(dim, int)
        return int((float(dim) - 1.0) * scale + 1.0) if dim % 2 else int(dim * scale)


    def prev_feature_resize(self, prev_feature, mode):
        if mode == 'down':
            feature_size_h = self.scale_dimension(prev_feature.shape[2], 0.5)
            feature_size_w = self.scale_dimension(prev_feature.shape[3], 0.5)
        elif mode == 'up':
            feature_size_h = self.scale_dimension(prev_feature.shape[2], 2)
            feature_size_w = self.scale_dimension(prev_feature.shape[3], 2)
        return F.interpolate(prev_feature, (feature_size_h, feature_size_w), mode='bilinear')


    def forward(self, s0, s1_down, s1_same, s1_up, n_alphas):
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
            for i in range(self.B):
                new_states = []
                for j, h in enumerate(states):
                    branch_index = offset + j
                    if self._ops[branch_index] is None:
                        continue
                    new_state = self._ops[branch_index](
                        h, n_alphas[branch_index])
                    new_states.append(new_state)

                s = sum(new_states)
                offset += len(states)
                states.append(s)

            concat_feature = torch.cat(states[-self.B:], dim=1)
            final_concates.append(concat_feature)

        return final_concates
