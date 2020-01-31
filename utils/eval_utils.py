import math
import random
import numpy as np
import torch
import torch.nn.functional as F

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

