import torch
import torch.nn as nn
from modeling.operations import *
from modeling.genotypes import PRIMITIVES
from modeling.genotypes import Genotype


x = torch.cuda.FloatTensor(10000, 500).normal_()
w = torch.cuda.FloatTensor(200, 500).normal_()


torch.cuda.synchronize()
torch.cuda.synchronize()

y = x.mm(w.t())
torch.cuda.synchronize() # wait for mm to finish

class measure_reduce(nn.Module):
    def __init__(self):
        super(measure_reduce, self).__init__()

        self.x1 = torch.cuda.FloatTensor(10000, 500).normal_()
        self.w1 = torch.cuda.FloatTensor(200, 500).normal_()
        torch.cuda.synchronize()
        torch.cuda.synchronize()

        self.ops=nn.ModuleList()
        self.ops.append(FactorizedReduce (20, 40, affine= False))
        self.ops.append(FactorizedReduce (40, 80, affine= False))
        self.ops.append(FactorizedReduce (80, 160, affine= False))

        self.x=[]
        self.x.append(torch.cuda.FloatTensor(10, 20, 128, 128).normal_())
        self.x.append(torch.cuda.FloatTensor(10, 40, 64, 64).normal_())
        self.x.append(torch.cuda.FloatTensor(10, 80, 32, 32).normal_())


    def forward(self):
        torch.cuda.synchronize() # wait for mm to finish
        time=0
        for j in range(3):
            for i in range(10000):
                y = self.x1.mm(self.w1.t())
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

                z = self.ops[j](self.x[j])

                end.record()
                torch.cuda.synchronize()
                if i>4999:
                    time+=start.elapsed_time(end)
            print(time/5000)
            print(self.ops[j])
            torch.cuda.synchronize()



class measure_increase(nn.Module):
    def __init__(self):
        super(measure_increase, self).__init__()

        self.x1 = torch.cuda.FloatTensor(10000, 500).normal_()
        self.w1 = torch.cuda.FloatTensor(200, 500).normal_()
        torch.cuda.synchronize()
        torch.cuda.synchronize()

        self.ops=nn.ModuleList()
        self.ops.append(FactorizedIncrease (160, 80))
        self.ops.append(FactorizedIncrease (80, 40))
        self.ops.append(FactorizedIncrease (40, 20))

        self.x=[]
        self.x.append(torch.cuda.FloatTensor(10, 160, 32, 32).normal_())
        self.x.append(torch.cuda.FloatTensor(10, 80, 64, 64).normal_())
        self.x.append(torch.cuda.FloatTensor(10, 40, 128, 128).normal_())


    def forward(self):
        torch.cuda.synchronize() # wait for mm to finish
        time=0
        for j in range(3):
            for i in range(10000):
                y = self.x1.mm(self.w1.t())
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

                z = self.ops[j](self.x[j])

                end.record()
                torch.cuda.synchronize()
                if i>4999:
                    time+=start.elapsed_time(end)
            print(time/5000)
            print(self.ops[j])
            torch.cuda.synchronize()


class measure_level(nn.Module):
    def __init__(self):
        super(measure_level, self).__init__()

        self.x1 = torch.cuda.FloatTensor(10000, 500).normal_()
        self.w1 = torch.cuda.FloatTensor(200, 500).normal_()
        torch.cuda.synchronize()
        torch.cuda.synchronize()

        self.ops=nn.ModuleList()
        self.ops.append(ReLUConvBN(20, 20, 1, 1, 0, affine=False))
        self.ops.append(ReLUConvBN(40, 40, 1, 1, 0, affine=False))
        self.ops.append(ReLUConvBN(80, 80, 1, 1, 0, affine=False))
        self.ops.append(ReLUConvBN(160, 160, 1, 1, 0, affine=False))


        self.ops.append(ReLUConvBN(100, 20, 1, 1, 0, affine=False))
        self.ops.append(ReLUConvBN(200, 40, 1, 1, 0, affine=False))
        self.ops.append(ReLUConvBN(400, 80, 1, 1, 0, affine=False))
        self.ops.append(ReLUConvBN(800, 160, 1, 1, 0, affine=False))

        self.x=[]
        self.x.append(torch.cuda.FloatTensor(10, 20, 128,128).normal_())
        self.x.append(torch.cuda.FloatTensor(10, 40, 64, 64).normal_())
        self.x.append(torch.cuda.FloatTensor(10, 80, 32, 32).normal_())
        self.x.append(torch.cuda.FloatTensor(10, 160, 16, 16).normal_())

        self.x.append(torch.cuda.FloatTensor(10, 100, 128, 128).normal_())
        self.x.append(torch.cuda.FloatTensor(10, 200, 64, 64).normal_())
        self.x.append(torch.cuda.FloatTensor(10, 400, 32, 32).normal_())
        self.x.append(torch.cuda.FloatTensor(10, 800, 16, 16).normal_())

    def forward(self):
        torch.cuda.synchronize() # wait for mm to finish
        time=0
        for j in range(8):
            for i in range(10000):
                y = self.x1.mm(self.w1.t())
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

                z = self.ops[j](self.x[j])

                end.record()
                torch.cuda.synchronize()
                if i>4999:
                    time+=start.elapsed_time(end)
            print(time/5000)
            print(self.ops[j])
            torch.cuda.synchronize()



model1=measure_reduce()
model1=model1.cuda()
model1.eval()
with torch.no_grad():
   model1()
print('************************')

model2=measure_increase()
model2=model2.cuda()
model2.eval()
with torch.no_grad():
   model2()
print('************************')

model3=measure_level()
model3=model3.cuda()
model3.eval()
with torch.no_grad():
   model3()
print('************************')
