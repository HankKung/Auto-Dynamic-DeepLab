import torch
import torch.nn as nn
from modeling.operations import *
from modeling.genotypes import PRIMITIVES
from modeling.genotypes import Genotype
import numpy as np

x = torch.cuda.FloatTensor(10000, 500).normal_()
w = torch.cuda.FloatTensor(200, 500).normal_()

torch.cuda.synchronize()
torch.cuda.synchronize()

y = x.mm(w.t())
torch.cuda.synchronize() # wait for mm to finish


class measure(nn.Module):
    def __init__(self, C):
        super(measure, self).__init__()
        self.C=C
        stride=1
        self.ops=nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](self.C, stride, False)
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
                 if i>4999:
                     start = torch.cuda.Event(enable_timing=True)
                     end = torch.cuda.Event(enable_timing=True)
                     start.record()
                     z=op(x)
                     end.record()
                     torch.cuda.synchronize()
                     time+=start.elapsed_time(end)
             print(time/5000)
             lat_dict[i][x.shape[1]] = time/5000

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


model1 = measure(20)
model1 = model1.cuda()
model1.eval()
with torch.no_grad():
    lat_dict = model1(x, lat_dict)
print('************************')
np.save('latency.npy', lat_dict) 
latency = np.load('latency.npy', allow_pickle='TRUE').item()
print(latency)

model2 = measure(40)
model2 = model2.cuda()
model2.eval()
with torch.no_grad():
    lat_dict = model2(x2, lat_dict)
print('************************')

model3 = measure(80)
model3 = model3.cuda()
model3.eval()
with torch.no_grad():
    lat_dict = model3(x3, lat_dict)
print('************************')

model4 = measure(160)
model4 = model4.cuda()
model4.eval()
with torch.no_grad():
    lat_dict = model4(x4, lat_dict)
np.save('latency.npy', lat_dict) 

