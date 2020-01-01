import torch
import torch.nn as nn
import numpy as np
from modeling import cell_level_search
from modeling.genotypes import PRIMITIVES
import torch.nn.functional as F
from modeling.operations import *
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class AutoDeeplab (nn.Module) :
    def __init__(self, num_classes, num_layers, F=8, B_d=4, B_c=5, 
                 distributed_layer=5, sync_bn=False ,cell=cell_level_search.Cell):
        super(AutoDeeplab, self).__init__()

        BatchNorm = SynchronizedBatchNorm2d if sync_bn == True else nn.BatchNorm2d
        self.cells = nn.ModuleList()
        self._num_layers = num_layers
        self._num_classes = num_classes
        self.B_d = B_d
        self.B_c = B_c
        self.distributed_layer = distributed_layer
        self._initialize_alphas_betas ()

        f_initial = F * B_d
        half_f_initial = int(f_initial / 2)

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, half_f_initial, 3, stride=2, padding=1),
            BatchNorm(half_f_initial),
            nn.ReLU ()
        )
        self.stem1 = nn.Sequential(
            nn.Conv2d(half_f_initial, f_initial, 3, stride=2, padding=1),
            BatchNorm(f_initial),
            nn.ReLU ()
        )

        FBD = F * B_d
        FBC = F * B_c
        for i in range (self._num_layers) :

            if i == 0 :
                cell1 = cell (B_d, half_f_initial,
                              None, f_initial, None,
                              F, BatchNorm=BatchNorm, pre_preprocess_sample_rate=0.5)
                cell2 = cell (B_d, half_f_initial,
                              f_initial, None, None,
                              F * 2, BatchNorm=BatchNorm, pre_preprocess_sample_rate=0.25)
                self.cells += [cell1]
                self.cells += [cell2]
            elif i == 1 :
                cell1 = cell (B_d, f_initial,
                              None, FBD, FBD * 2,
                              F, BatchNorm=BatchNorm)

                cell2 = cell (B_d, f_initial,
                              FBD, FBD * 2, None,
                              F * 2, BatchNorm=BatchNorm, pre_preprocess_sample_rate=0.5)

                cell3 = cell (B_d, f_initial,
                              FBD * 2, None, None,
                              F * 4, BatchNorm=BatchNorm, pre_preprocess_sample_rate=0.25)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]

            elif i == 2 :
                cell1 = cell (B_d, FBD,
                              None, FBD, FBD * 2,
                              F, BatchNorm=BatchNorm)

                cell2 = cell (B_d, FBD * 2,
                              FBD, FBD * 2, FBD * 4,
                              F * 2, BatchNorm=BatchNorm)

                cell3 = cell (B_d, FBD * 2,
                              FBD * 2, FBD * 4, None,
                              F * 4, BatchNorm=BatchNorm, pre_preprocess_sample_rate=0.5)

                cell4 = cell (B_d, FBD * 2,
                              FBD * 4, None, None,
                              F * 8, BatchNorm=BatchNorm, pre_preprocess_sample_rate=0.25)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]

            elif i == 3 :
                cell1 = cell (B_d, FBD,
                              None, FBD, FBD * 2,
                              F, BatchNorm=BatchNorm)

                cell2 = cell (B_d, FBD * 2,
                              FBD, FBD * 2, FBD * 4,
                              F * 2, BatchNorm=BatchNorm)

                cell3 = cell (B_d, FBD * 4,
                              FBD * 2, FBD * 4, FBD * 8,
                              F * 4, BatchNorm=BatchNorm)


                cell4 = cell (B_d, FBD * 4,
                              FBD * 4, FBD * 8, None,
                              F * 8, BatchNorm=BatchNorm, pre_preprocess_sample_rate=0.5)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]

            elif i < distributed_layer :
                cell1 = cell (B_d, FBD,
                              None, FBD, FBD * 2,
                              F, BatchNorm=BatchNorm)

                cell2 = cell (B_d, FBD * 2,
                              FBD, FBD * 2, FBD * 4,
                              F * 2, BatchNorm=BatchNorm)

                cell3 = cell (B_d, FBD * 4,
                              FBD * 2, FBD * 4, FBD * 8,
                              F * 4, BatchNorm=BatchNorm)

                cell4 = cell (B_d, FBD * 8,
                              FBD * 4, FBD * 8, None,
                              F * 8, BatchNorm=BatchNorm)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]

            elif i == distributed_layer:
                cell1 = cell (B_d, FBD,
                              None, FBD, FBD * 2,
                              F, BatchNorm=BatchNorm)

                cell2 = cell (B_d, FBD * 2,
                              FBD, FBD * 2, FBD * 4,
                              F * 2, BatchNorm=BatchNorm)

                cell3 = cell (B_d, FBD * 4,
                              FBD * 2, FBD * 4, FBD * 8,
                              F * 4, BatchNorm=BatchNorm)

                cell4 = cell (B_d, FBD * 8,
                              FBD * 4, FBD * 8, None,
                              F * 8, BatchNorm=BatchNorm)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]

            elif i == distributed_layer+1:
                cell1 = cell (B_c, FBD,
                              None, FBD, FBD * 2,
                              F, BatchNorm=BatchNorm)

                cell2 = cell (B_c, FBD * 2,
                              FBD, FBD * 2, FBD * 4,
                              F * 2, BatchNorm=BatchNorm)

                cell3 = cell (B_c, FBD * 4,
                              FBD * 2, FBD * 4, FBD * 8,
                              F * 4, BatchNorm=BatchNorm)

                cell4 = cell (B_c, FBD * 8,
                              FBD * 4, FBD * 8, None,
                              F * 8, BatchNorm=BatchNorm)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]

            elif i == distributed_layer+2:
                cell1 = cell (B_c, FBD,
                              None, FBC, FBC * 2,
                              F, BatchNorm=BatchNorm)

                cell2 = cell (B_c, FBD * 2,
                              FBC, FBC * 2, FBC * 4,
                              F * 2, BatchNorm=BatchNorm)

                cell3 = cell (B_c, FBD * 4,
                              FBC * 2, FBC * 4, FBC * 8,
                              F * 4, BatchNorm=BatchNorm)

                cell4 = cell (B_c, FBD * 8,
                              FBC * 4, FBC * 8, None,
                              F * 8, BatchNorm=BatchNorm)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]

            else :
                cell1 = cell (B_c, FBC,
                              None, FBC, FBC * 2,
                              F, BatchNorm=BatchNorm)

                cell2 = cell (B_c, FBC * 2,
                              FBC, FBC * 2, FBC * 4,
                              F * 2, BatchNorm=BatchNorm)

                cell3 = cell (B_c, FBC * 4,
                              FBC * 2, FBC * 4, FBC * 8,
                              F * 4, BatchNorm=BatchNorm)

                cell4 = cell (B_c, FBC * 8,
                              FBC * 4, FBC * 8, None,
                              F * 8, BatchNorm=BatchNorm)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]

        self.aspp_device_4 = nn.Sequential (
            ASPP (FBD, self._num_classes, 24, 24, BatchNorm=BatchNorm) #96 / 4 as in the paper
        )
        self.aspp_device_8 = nn.Sequential (
            ASPP (FBD * 2, self._num_classes, 12, 12, BatchNorm=BatchNorm) #96 / 8
        )
        self.aspp_device_16 = nn.Sequential (
            ASPP (FBD * 4, self._num_classes, 6, 6, BatchNorm=BatchNorm) #96 / 16
        )
        self.aspp_device_32 = nn.Sequential (
            ASPP (FBD * 8, self._num_classes, 3, 3, BatchNorm=BatchNorm) #96 / 32
        )


        self.aspp_4 = nn.Sequential (
            ASPP (FBC, self._num_classes, 24, 24, BatchNorm=BatchNorm) #96 / 4 as in the paper
        )
        self.aspp_8 = nn.Sequential (
            ASPP (FBC * 2, self._num_classes, 12, 12, BatchNorm=BatchNorm) #96 / 8
        )
        self.aspp_16 = nn.Sequential (
            ASPP (FBC * 4, self._num_classes, 6, 6, BatchNorm=BatchNorm) #96 / 16
        )
        self.aspp_32 = nn.Sequential (
            ASPP (FBC * 8, self._num_classes, 3, 3, BatchNorm=BatchNorm) #96 / 32
        )
        self._init_weight()

    def forward (self, x) :
        self.level_4 = []
        self.level_8 = []
        self.level_16 = []
        self.level_32 = []
        temp = self.stem0 (x)
        self.level_4.append (self.stem1 (temp))

        count = 0

        normalized_betas = torch.randn(12, 4, 3).cuda().half()
        # Softmax on alphas and betas
        if torch.cuda.device_count() > 1:
            img_device = torch.device('cuda', x.get_device())
            normalized_alphas_d = F.softmax(self.alphas_d.to(device=img_device), dim=-1)
            normalized_alphas_c = F.softmax(self.alphas_c.to(device=img_device), dim=-1)
            
            # normalized_betas[layer][ith node][0 : ➚, 1: ➙, 2 : ➘]
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
            normalized_alphas_d = F.softmax(self.alphas_d, dim=-1)
            normalized_alphas_c = F.softmax(self.alphas_c, dim=-1)

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
                level4_new, = self.cells[count] (temp, None, self.level_4[-1], None, normalized_alphas_d)
                count += 1
                level8_new, = self.cells[count] (temp, self.level_4[-1], None, None, normalized_alphas_d)
                count += 1

                level4_new = normalized_betas[layer][0][1] * level4_new
                level8_new = normalized_betas[layer][0][2] * level8_new
                
                self.level_4.append (level4_new)
                self.level_8.append (level8_new)
                del temp

            elif layer == 1 :
                level4_new_1, level4_new_2 = self.cells[count] (self.level_4[-2],
                                                                None,
                                                                self.level_4[-1],
                                                                self.level_8[-1],
                                                                normalized_alphas_d)
                count += 1
                level4_new = normalized_betas[layer][0][1] * level4_new_1 + normalized_betas[layer][1][0] * level4_new_2

                level8_new_1, level8_new_2 = self.cells[count] (self.level_4[-2],
                                                                self.level_4[-1],
                                                                self.level_8[-1],
                                                                None,
                                                                normalized_alphas_d)
                count += 1
                level8_new = normalized_betas[layer][0][2] * level8_new_1 + normalized_betas[layer][1][1] * level8_new_2

                level16_new, = self.cells[count] (self.level_4[-2],
                                                  self.level_8[-1],
                                                  None,
                                                  None,
                                                  normalized_alphas_d)
                level16_new = normalized_betas[layer][1][2] * level16_new
                count += 1


                self.level_4.append (level4_new)
                self.level_8.append (level8_new)
                self.level_16.append (level16_new)

            elif layer == 2 :
                level4_new_1, level4_new_2 = self.cells[count] (self.level_4[-2],
                                                                None,
                                                                self.level_4[-1],
                                                                self.level_8[-1],
                                                                normalized_alphas_d)
                count += 1
                level4_new = normalized_betas[layer][0][1] * level4_new_1 + normalized_betas[layer][1][0] * level4_new_2

                level8_new_1, level8_new_2, level8_new_3 = self.cells[count] (self.level_8[-2],
                                                                              self.level_4[-1],
                                                                              self.level_8[-1],
                                                                              self.level_16[-1],
                                                                              normalized_alphas_d)
                count += 1
                level8_new = normalized_betas[layer][0][2] * level8_new_1 + normalized_betas[layer][1][1] * level8_new_2 + normalized_betas[layer][2][0] * level8_new_3

                level16_new_1, level16_new_2 = self.cells[count] (self.level_8[-2],
                                                                  self.level_8[-1],
                                                                  self.level_16[-1],
                                                                  None,
                                                                  normalized_alphas_d)
                count += 1
                level16_new = normalized_betas[layer][1][2] * level16_new_1 + normalized_betas[layer][2][1] * level16_new_2


                level32_new, = self.cells[count] (self.level_8[-2],
                                                  self.level_16[-1],
                                                  None,
                                                  None,
                                                  normalized_alphas_d)
                level32_new = normalized_betas[layer][2][2] * level32_new
                count += 1

                self.level_4.append (level4_new)
                self.level_8.append (level8_new)
                self.level_16.append (level16_new)
                self.level_32.append (level32_new)

            elif layer == 3 :
                level4_new_1, level4_new_2 = self.cells[count] (self.level_4[-2],
                                                                None,
                                                                self.level_4[-1],
                                                                self.level_8[-1],
                                                                normalized_alphas_d)
                count += 1
                level4_new = normalized_betas[layer][0][1] * level4_new_1 + normalized_betas[layer][1][0] * level4_new_2

                level8_new_1, level8_new_2, level8_new_3 = self.cells[count] (self.level_8[-2],
                                                                              self.level_4[-1],
                                                                              self.level_8[-1],
                                                                              self.level_16[-1],
                                                                              normalized_alphas_d)
                count += 1
                level8_new = normalized_betas[layer][0][2] * level8_new_1 + normalized_betas[layer][1][1] * level8_new_2 + normalized_betas[layer][2][0] * level8_new_3

                level16_new_1, level16_new_2, level16_new_3 = self.cells[count] (self.level_16[-2],
                                                                                 self.level_8[-1],
                                                                                 self.level_16[-1],
                                                                                 self.level_32[-1],
                                                                                 normalized_alphas_d)
                count += 1
                level16_new = normalized_betas[layer][1][2] * level16_new_1 + normalized_betas[layer][2][1] * level16_new_2 + normalized_betas[layer][3][0] * level16_new_3


                level32_new_1, level32_new_2 = self.cells[count] (self.level_16[-2],
                                                                  self.level_16[-1],
                                                                  self.level_32[-1],
                                                                  None,
                                                                  normalized_alphas_d)
                count += 1
                level32_new = normalized_betas[layer][2][2] * level32_new_1 + normalized_betas[layer][3][1] * level32_new_2


                self.level_4.append (level4_new)
                self.level_8.append (level8_new)
                self.level_16.append (level16_new)
                self.level_32.append (level32_new)


            elif layer < self.distributed_layer :
                level4_new_1, level4_new_2 = self.cells[count] (self.level_4[-2],
                                                                None,
                                                                self.level_4[-1],
                                                                self.level_8[-1],
                                                                normalized_alphas_d)
                count += 1
                level4_new = normalized_betas[layer][0][1] * level4_new_1 + normalized_betas[layer][1][0] * level4_new_2

                level8_new_1, level8_new_2, level8_new_3 = self.cells[count] (self.level_8[-2],
                                                                              self.level_4[-1],
                                                                              self.level_8[-1],
                                                                              self.level_16[-1],
                                                                              normalized_alphas_d)
                count += 1
                level8_new = normalized_betas[layer][0][2] * level8_new_1 + normalized_betas[layer][1][1] * level8_new_2 + normalized_betas[layer][2][0] * level8_new_3

                level16_new_1, level16_new_2, level16_new_3 = self.cells[count] (self.level_16[-2],
                                                                                 self.level_8[-1],
                                                                                 self.level_16[-1],
                                                                                 self.level_32[-1],
                                                                                 normalized_alphas_d)
                count += 1
                level16_new = normalized_betas[layer][1][2] * level16_new_1 + normalized_betas[layer][2][1] * level16_new_2 + normalized_betas[layer][3][0] * level16_new_3


                level32_new_1, level32_new_2 = self.cells[count] (self.level_32[-2],
                                                                  self.level_16[-1],
                                                                  self.level_32[-1],
                                                                  None,
                                                                  normalized_alphas_d)
                count += 1
                level32_new = normalized_betas[layer][2][2] * level32_new_1 + normalized_betas[layer][3][1] * level32_new_2


                self.level_4.append (level4_new)
                self.level_8.append (level8_new)
                self.level_16.append (level16_new)
                self.level_32.append (level32_new)

            elif layer == self.distributed_layer:
                level4_new_1, level4_new_2 = self.cells[count] (self.level_4[-2],
                                                                None,
                                                                self.level_4[-1],
                                                                self.level_8[-1],
                                                                normalized_alphas_d)
                count += 1
                level4_new = normalized_betas[layer][0][1] * level4_new_1 + normalized_betas[layer][1][0] * level4_new_2
                
                level8_new_1, level8_new_2, level8_new_3 = self.cells[count] (self.level_8[-2],
                                                                              self.level_4[-1],
                                                                              self.level_8[-1],
                                                                              self.level_16[-1],
                                                                              normalized_alphas_d)
                count += 1

                level8_new = normalized_betas[layer][0][2] * level8_new_1 + normalized_betas[layer][1][1] * level8_new_2 + normalized_betas[layer][2][0] * level8_new_3

                level16_new_1, level16_new_2, level16_new_3 = self.cells[count] (self.level_16[-2],
                                                                                 self.level_8[-1],
                                                                                 self.level_16[-1],
                                                                                 self.level_32[-1],
                                                                                 normalized_alphas_d)
                count += 1
                level16_new = normalized_betas[layer][1][2] * level16_new_1 + normalized_betas[layer][2][1] * level16_new_2 + normalized_betas[layer][3][0] * level16_new_3
                
                level32_new_1, level32_new_2 = self.cells[count] (self.level_32[-2],
                                                                  self.level_16[-1],
                                                                  self.level_32[-1],
                                                                  None,
                                                                  normalized_alphas_d)
                count += 1
                level32_new = normalized_betas[layer][2][2] * level32_new_1 + normalized_betas[layer][3][1] * level32_new_2


                self.level_4.append (level4_new)
                self.level_8.append (level8_new)
                self.level_16.append (level16_new)
                self.level_32.append (level32_new)
                device_4_new = self.aspp_device_4(self.level_4[-1])
                device_8_new = self.aspp_device_8(self.level_8[-1])
                device_16_new = self.aspp_device_16(self.level_16[-1])
                device_32_new = self.aspp_device_32(self.level_32[-1])

            else :
                level4_new_1, level4_new_2 = self.cells[count] (self.level_4[-2],
                                                                None,
                                                                self.level_4[-1],
                                                                self.level_8[-1],
                                                                normalized_alphas_c)
                count += 1
                level4_new = normalized_betas[layer][0][1] * level4_new_1 + normalized_betas[layer][1][0] * level4_new_2

                level8_new_1, level8_new_2, level8_new_3 = self.cells[count] (self.level_8[-2],
                                                                              self.level_4[-1],
                                                                              self.level_8[-1],
                                                                              self.level_16[-1],
                                                                              normalized_alphas_c)
                count += 1

                level8_new = normalized_betas[layer][0][2] * level8_new_1 + normalized_betas[layer][1][1] * level8_new_2 + normalized_betas[layer][2][0] * level8_new_3

                level16_new_1, level16_new_2, level16_new_3 = self.cells[count] (self.level_16[-2],
                                                                                 self.level_8[-1],
                                                                                 self.level_16[-1],
                                                                                 self.level_32[-1],
                                                                                 normalized_alphas_c)
                count += 1
                level16_new = normalized_betas[layer][1][2] * level16_new_1 + normalized_betas[layer][2][1] * level16_new_2 + normalized_betas[layer][3][0] * level16_new_3


                level32_new_1, level32_new_2 = self.cells[count] (self.level_32[-2],
                                                                  self.level_16[-1],
                                                                  self.level_32[-1],
                                                                  None,
                                                                  normalized_alphas_c)
                count += 1
                level32_new = normalized_betas[layer][2][2] * level32_new_1 + normalized_betas[layer][3][1] * level32_new_2


                self.level_4.append (level4_new)
                self.level_8.append (level8_new)
                self.level_16.append (level16_new)
                self.level_32.append (level32_new)

            self.level_4 = self.level_4[-2:]
            self.level_8 = self.level_8[-2:]
            self.level_16 = self.level_16[-2:]
            self.level_32 = self.level_32[-2:]

        aspp_result_4 = self.aspp_4 (self.level_4[-1])
        del self.level_4
        aspp_result_8 = self.aspp_8 (self.level_8[-1])
        del self.level_8
        aspp_result_16 = self.aspp_16 (self.level_16[-1])
        del self.level_16
        aspp_result_32 = self.aspp_32 (self.level_32[-1])
        del self.level_32
        upsample = nn.Upsample(size=x.size()[2:], mode='bilinear', align_corners=True)
        aspp_result_4 = upsample (aspp_result_4)
        aspp_result_8 = upsample (aspp_result_8)
        aspp_result_16 = upsample (aspp_result_16)
        aspp_result_32 = upsample (aspp_result_32)

        device_4_new = upsample(device_4_new)
        device_8_new = upsample(device_8_new)
        device_16_new = upsample(device_16_new)
        device_32_new = upsample(device_32_new)

        sum_device_feature_map = device_4_new + device_8_new + device_16_new + device_32_new
        del device_4_new, device_8_new, device_16_new, device_32_new
        sum_feature_map = aspp_result_4 + aspp_result_8 + aspp_result_16 + aspp_result_32

        return sum_device_feature_map, sum_feature_map

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

    def _initialize_alphas_betas(self):
        k_d = sum(1 for i in range(self._step_d) for n in range(2+i))
        k_c = sum(1 for i in range(self._step_c) for n in range(2+i))
        num_ops = len(PRIMITIVES)
        alphas_d = torch.tensor (1e-3*torch.randn(k_d, num_ops).cuda(), requires_grad=True)
        alphas_c = torch.tensor (1e-3*torch.randn(k_c, num_ops).cuda(), requires_grad=True)
        betas = torch.tensor (1e-3*torch.randn(12, 4, 3).cuda(), requires_grad=True)

        self._arch_parameters = [
            alphas_d,
            alphas_c,
            betas,
            ]
        self._arch_param_names = [
            'alphas_d',
            'alphas_c',
            'betas',
            ]

        [self.register_parameter(name, torch.nn.Parameter(param)) for name, param in zip(self._arch_param_names, self._arch_parameters)]

    def arch_parameters (self) :
        return [param for name, param in self.named_parameters() if name in self._arch_param_names]

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if name not in self._arch_param_names]



def main () :
    model = AutoDeeplab (7, 12, None)
    x = torch.tensor (torch.ones (4, 3, 224, 224))

if __name__ == '__main__' :
    main ()
