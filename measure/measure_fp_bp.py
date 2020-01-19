#ref: https://gist.github.com/iacolippo/9611c6d9c7dfc469314baeb5a69e7e1b

import gc
import numpy as np
import sys
import time
import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.backends.cudnn as cudnn


def measure(model, x, y):
    # synchronize gpu time and measure fp
    torch.cuda.synchronize()
    t0 = time.time()
    y_pred = model(x)
    torch.cuda.synchronize()
    elapsed_fp = time.time()-t0
    
    # zero gradients, synchronize time and measure
    model.zero_grad()
    t0 = time.time()
    y_pred.backward(y)
    torch.cuda.synchronize()
    elapsed_bp = time.time()-t0
    return elapsed_fp, elapsed_bp

def benchmark(model, x, y):
    # transfer the model on GPU
    model.cuda()
    
    # DRY RUNS
    for i in range(5):
        _, _ = measure(model, x, y)

    print('DONE WITH DRY RUNS, NOW BENCHMARKING')

    # START BENCHMARKING
    t_forward = []
    t_backward = []
    for i in range(10):
        t_fp, t_bp = measure(model, x, y)
        t_forward.append(t_fp)
        t_backward.append(t_bp)
    
    # free memory
    del model
    
    return t_forward, t_backward

def main():
    # set the seed for RNG
    if len(sys.argv)==2:
        torch.manual_seed(int(sys.argv[1]))
    else:
        torch.manual_seed(1234)

    # set cudnn backend to benchmark config
    cudnn.benchmark = True
    
    # instantiate the models
    resnet18 = models.resnet18()
    resnet34 = models.resnet34()
    resnet50 = models.resnet50()
    resnet101 = models.resnet101()
    resnet152 = models.resnet152()
    alexnet = models.alexnet()
    vgg16 = models.vgg16()
    
    # build the dict to iterate over
    architectures = {'resnet18': resnet18, 
                     'resnet34': resnet34,
                     'resnet50': resnet50,
                     'resnet101': resnet101,
                     'resnet152': resnet152,
                     'alexnet': alexnet, 
                     'vgg16': vgg16
                     }
    
    # build dummy variables to input and output
    x = Variable(torch.randn(1, 3, 224, 224)).cuda()
    y = torch.randn(1, 1000).cuda()

    # loop over architectures and measure them
    for deep_net in architectures:
        print(deep_net)
        t_fp, t_bp = benchmark(architectures[deep_net], x, y)
        # print results
        print('FORWARD PASS: ', np.mean(np.asarray(t_fp)*1e3), '+/-', np.std(np.asarray(t_fp)*1e3))
        print('BACKWARD PASS: ', np.mean(np.asarray(t_bp)*1e3), '+/-', np.std(np.asarray(t_bp)*1e3))
        print('RATIO BP/FP:', np.mean(np.asarray(t_bp))/np.mean(np.asarray(t_fp)))
        
        # write the list of measures in files
        fname = deep_net+'-benchmark.txt'
        with open(fname, 'w') as f:
            for (fp_time, bp_time) in zip(t_fp, t_bp):
                f.write(str(fp_time)+" "+str(bp_time)+" \n")
        
        # force garbage collection
        gc.collect()

if __name__ == '__main__':
    main()