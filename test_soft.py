import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

betas = torch.tensor (torch.randn(12, 4, 3).cuda(), requires_grad=True)
print(betas[-2:])
normalized_betas = torch.zeros(12, 4, 3)
for layer in range (len(betas)):
    if layer == 0:
        normalized_betas[layer][0][1:] = F.softmax (betas[layer][0][1:], dim=-1)

    elif layer == 1:
        normalized_betas[layer][0][1:] = F.softmax (betas[layer][0][1:], dim=-1)
        normalized_betas[layer][1] = F.softmax (betas[layer][1], dim=-1)

    elif layer == 2:
        normalized_betas[layer][0][1:] = F.softmax (betas[layer][0][1:], dim=-1)
        normalized_betas[layer][1] = F.softmax (betas[layer][1], dim=-1)
        normalized_betas[layer][2] = F.softmax (betas[layer][2], dim=-1)
    else :
        betas[layer][0][1:] /= min(betas[layer][0][1:])
        betas[layer][1] /= min(betas[layer][1])
        betas[layer][2] /= min(betas[layer][2])
        betas[layer][3][:2] /= min(betas[layer][3][:2])
        normalized_betas[layer][0][1:] = F.softmax (betas[layer][0][1:], dim=-1)
        normalized_betas[layer][1] = F.softmax (betas[layer][1], dim=-1)
        normalized_betas[layer][2] = F.softmax (betas[layer][2], dim=-1)
        normalized_betas[layer][3][:2] = F.softmax (betas[layer][3][:2], dim=-1)
print(normalized_betas[-2:])


#a= torch.tensor(1e-3*torch.randn(3).cuda())
#m = min(a)
#a /= m
#b = a[1:]
#print(a)
#print(F.softmax(a, dim=-1))
#print(F.softmax(a[1:], dim=-1))
#print(F.softmax(b, dim= -1))
