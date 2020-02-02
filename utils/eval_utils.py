import math
import random
import numpy as np
import torch
import torch.nn.functional as F


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "aux" not in name)/1e6


def encoding_alphas(genotype):
    k = sum(1 for i in range(5) for n in range(2+i))
    alphas = torch.tensor.zeros(torch.randn(k, 8).cuda())

    for i in range(len(genotype)):
        alphas[genotype[i][0]][genotype[i][1]] = 1
    return alphas

def encoding_betas(beta):
    betas = beta * 10000

    for layer in range (len(betas)):
        if layer == 0:
            normalized_betas[layer][0][1:] = F.softmax (betas[layer][0][1:], dim=-1) * (2/3)

        elif layer == 1:
            normalized_betas[layer][0][1:] = F.softmax (betas[layer][0][1:], dim=-1) * (2/3)
            normalized_betas[layer][1] = F.softmax (betas[layer][1], dim=-1)

        elif layer == 2:
            normalized_betas[layer][0][1:] = F.softmax (betas[layer][0][1:], dim=-1) * (2/3)
            normalized_betas[layer][1] = F.softmax (betas[layer][1], dim=-1)
            normalized_betas[layer][2] = F.softmax (betas[layer][2], dim=-1)
        else :
            normalized_betas[layer][0][1:] = F.softmax (betas[layer][0][1:], dim=-1) * (2/3)
            normalized_betas[layer][1] = F.softmax (betas[layer][1], dim=-1)
            normalized_betas[layer][2] = F.softmax (betas[layer][2], dim=-1)
            normalized_betas[layer][3][:2] = F.softmax (betas[layer][3][:2], dim=-1) * (2/3)

    return normalized_betas

