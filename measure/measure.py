import torch
import torch.nn as nn
from modeling.operations import *
from modeling.genotypes import PRIMITIVES
from modeling.genotypes import Genotype
from modeling.genotypes import aspp_train

import numpy as np
import torch.backends.cudnn as cudnn
torch.backends.cudnn.benchmark = True


x = torch.cuda.FloatTensor(10000, 500).normal_()
w = torch.cuda.FloatTensor(200, 500).normal_()

torch.cuda.synchronize()
torch.cuda.synchronize()

y = x.mm(w.t())
torch.cuda.synchronize() # wait for mm to finish


class measure_op(nn.Module):
    def __init__(self, C):
        super(measure_op, self).__init__()
        self.C=C
        stride=1
        eps=1e-5
        momentum=0.1
        self.ops=nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](self.C, stride, nn.BatchNorm2d, eps, momentum, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self.ops.append(op)

        self.x1 = torch.cuda.FloatTensor(10000, 500).normal_()
        self.w1 = torch.cuda.FloatTensor(200, 500).normal_()
        torch.cuda.synchronize()
        torch.cuda.synchronize()


    def forward(self, x, lat_dict):
        for op_i, op in enumerate(self.ops):
            y = self.x1.mm(self.w1.t())
            torch.cuda.synchronize()
            time=0
            for i in range(10000):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                torch.cuda.synchronize()

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                z=op(x)
                end.record()
                torch.cuda.synchronize()
                if i>4999:
                    time+=start.elapsed_time(end)
            print(time/5000)
            lat_dict[op_i][x.shape[1]] = time/5000

            print(op)
            torch.cuda.synchronize()
        return lat_dict

def scale_dimension(dim, scale):
    return int((float(dim) - 1.0) * scale + 1.0)

h = 1025
w = 2049

h1 = scale_dimension(h, 0.25)
w1 = scale_dimension(w, 0.25)
x1 = torch.cuda.FloatTensor(1, 20, h1, w1).normal_()
x1=x1.cuda()

h2 = scale_dimension(h, 0.125)
w2 = scale_dimension(w, 0.125)
x2 = torch.cuda.FloatTensor(1, 40, h2, w2).normal_()
x2=x2.cuda()

h3 = scale_dimension(h, 0.0625)
w3 = scale_dimension(w, 0.0625)
x3 = torch.cuda.FloatTensor(1, 80, h3, w3).normal_()
x3=x3.cuda()

h4 = scale_dimension(h, 0.03125)
w4 = scale_dimension(w, 0.03125)
x4 = torch.cuda.FloatTensor(1, 160, h4, w4).normal_()
x4=x4.cuda()


lat_dict = dict()
for i in range(8):
    lat_dict[i] = dict()


model1 = measure_op(20)
model1 = model1.cuda()
model1.eval()
with torch.no_grad():
    lat_dict = model1(x1, lat_dict)
print('************************************************************************')


model2 = measure_op(40)
model2 = model2.cuda()
model2.eval()
with torch.no_grad():
    lat_dict = model2(x2, lat_dict)
print('************************************************************************')


model3 = measure_op(80)
model3 = model3.cuda()
model3.eval()
with torch.no_grad():
    lat_dict = model3(x3, lat_dict)
print('************************************************************************')


model4 = measure_op(160)
model4 = model4.cuda()
model4.eval()
with torch.no_grad():
    lat_dict = model4(x4, lat_dict)
np.save('latency_op.npy', lat_dict) 


print('************************************************************************')

aspp_dict = dict()
x1 = torch.cuda.FloatTensor(1, 100, h1, w1).normal_()
x1=x1.cuda()

h2 = scale_dimension(h, 0.125)
w2 = scale_dimension(w, 0.125)
x2 = torch.cuda.FloatTensor(1, 200, h2, w2).normal_()
x2=x2.cuda()

h3 = scale_dimension(h, 0.0625)
w3 = scale_dimension(w, 0.0625)
x3 = torch.cuda.FloatTensor(1, 400, h3, w3).normal_()
x3=x3.cuda()

h4 = scale_dimension(h, 0.03125)
w4 = scale_dimension(w, 0.03125)
x4 = torch.cuda.FloatTensor(1, 800, h4, w4).normal_()
x4=x4.cuda()

aspp_0 = nn.Sequential (
            ASPP (100, 256, 19, 4, BatchNorm=nn.BatchNorm2d)).cuda()
aspp_1 = nn.Sequential (
            ASPP (200, 256, 19, 2, BatchNorm=nn.BatchNorm2d)).cuda()
aspp_2 = nn.Sequential (
            ASPP (400, 256 ,19, 1, BatchNorm=nn.BatchNorm2d)).cuda()
aspp_3 = nn.Sequential (
            ASPP (800, 256, 19, 0.5, BatchNorm=nn.BatchNorm2d)).cuda()
aspp_0.eval()
aspp_1.eval()
aspp_2.eval()
aspp_3.eval()

x = torch.cuda.FloatTensor(10000, 500).normal_()
w = torch.cuda.FloatTensor(200, 500).normal_()

torch.cuda.synchronize()
torch.cuda.synchronize()

y = x.mm(w.t())
torch.cuda.synchronize() # wait for mm to finish


with torch.no_grad():
    for i in range(10000):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        aspp_0(x0)
        end.record()
        torch.cuda.synchronize()
        if i >4999:
            time+=start.elapsed_time(end)
        torch.cuda.synchronize()
    aspp_1[0] = time/5000

with torch.no_grad():
    for i in range(10000):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        aspp_1(x1)
        end.record()
        torch.cuda.synchronize()
        if i >4999:
            time+=start.elapsed_time(end)
        torch.cuda.synchronize()
    aspp_1[1] = time/5000

with torch.no_grad():
    for i in range(10000):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        aspp_2(x2)
        end.record()
        torch.cuda.synchronize()
        if i >4999:
            time+=start.elapsed_time(end)
        torch.cuda.synchronize()
    aspp_1[2] = time/5000

with torch.no_grad():
    for i in range(10000):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        aspp_3(x3)
        end.record()
        torch.cuda.synchronize()
        if i >4999:
            time+=start.elapsed_time(end)
        torch.cuda.synchronize()
    aspp_1[3] = time/5000

np.save('latency_aspp.npy', aspp_dict) 

