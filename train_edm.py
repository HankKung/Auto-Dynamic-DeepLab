import argparse
import os
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torchviz import make_dot, make_dot_from_trace

from mypath import Path
from dataloaders import make_data_loader

from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from utils.copy_state_dict import copy_state_dict
from utils.eval_utils import AverageMeter
# from utils.encoding import *

from modeling.baseline_model import *
# from modeling.ADD import *
from modeling.ADD import *
from modeling.operations import normalized_shannon_entropy
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.sync_batchnorm.replicate import patch_replication_callback

from apex import amp
from ptflops import get_model_complexity_info
from torch.utils.data import TensorDataset, DataLoader


torch.backends.cudnn.benchmark = True


class trainNew(object):
    def __init__(self, args):
        self.args = args

        """ Define Saver """
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        """ Define Tensorboard Summary """
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()


        """ Define Dataloader """
        kwargs = {'num_workers': args.workers, 'pin_memory': True, 'drop_last': True}
        self.train_loader, self.val_loader, _, self.nclass = make_data_loader(args, **kwargs)


        self.criterion = nn.L1Loss()
        if args.network == 'searched-dense':
            cell_path = os.path.join(args.saved_arch_path, 'autodeeplab', 'genotype.npy')
            cell_arch = np.load(cell_path)

            if self.args.C == 2:
                C_index = [5]
                network_arch = [1, 2, 2, 2, 3, 2, 2, 1, 1, 1, 1, 2]
                low_level_layer = 0
            elif self.args.C == 3:
                C_index = [3, 7]
                network_arch = [1, 2, 3, 2, 2, 3, 2, 3, 2, 3, 2, 3]
                low_level_layer = 0
            elif self.args.C == 4:
                C_index = [2, 5, 8]
                network_arch = [1, 2, 3, 3, 2, 3, 3, 3, 3, 3, 2, 2]
                low_level_layer = 0

            model = ADD(network_arch,
                            C_index,
                            cell_arch,
                            self.nclass,
                            args,
                            low_level_layer)

        elif args.network.startswith('autodeeplab'):
            network_arch = [0, 0, 0, 1, 2, 1, 2, 2, 3, 3, 2, 1]
            cell_path = os.path.join(args.saved_arch_path, 'autodeeplab', 'genotype.npy')
            cell_arch = np.load(cell_path)
            low_level_layer = 2
            if self.args.C == 2:
                C_index = [5]
            elif self.args.C == 3:
                C_index = [3, 7]
            elif self.args.C == 4:
                C_index = [2, 5, 8]

            if args.network == 'autodeeplab-dense':
                model = ADD(network_arch,
                            C_index,
                            cell_arch,
                            self.nclass,
                            args,
                            low_level_layer)

            elif args.network == 'autodeeplab-baseline':
                model = Baselin_Model(network_arch,
                                    C_index,
                                    cell_arch,
                                    self.nclass,
                                    args,
                                    low_level_layer)

        self.edm = EDM().cuda()
        optimizer = torch.optim.Adam(self.edm.parameters(), lr=args.lr)
        self.model, self.optimizer = model, optimizer
        
        if args.cuda:
            self.model = self.model.cuda()

        """ Resuming checkpoint """
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']

            """ if the weights are wrapped in module object we have to clean it """
            if args.clean_module:
                self.model.load_state_dict(checkpoint['state_dict'])
                state_dict = checkpoint['state_dict']
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove 'module.' of dataparallel
                    new_state_dict[name] = v
                copy_state_dict(self.model.state_dict(), new_state_dict)

            else:
                if (torch.cuda.device_count() > 1):
                    copy_state_dict(self.model.module.state_dict(), checkpoint['state_dict'])
                else:
                    copy_state_dict(self.model.state_dict(), checkpoint['state_dict'])

        if os.path.isfile('feature.npy'):
            train_feature = np.load('feature.npy')
            train_entropy = np.load('entropy.npy')
            train_set = TensorDataset(torch.tensor(train_feature), torch.tensor(train_entropy, dtype=torch.float))
            train_set = DataLoader(train_set, batch_size=self.args.train_batch, shuffle=True, pin_memory=True)
            self.train_set = train_set
        else:
            self.make_data(self.args.train_batch)

    def make_data(self, batch_size):
        self.model.eval()
        tbar = tqdm(self.train_loader, desc='\r')
        train_feature = []
        train_entropy = []
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            with torch.no_grad():
                output, feature = self.model.get_feature(image)
                train_entropy.append(normalized_shannon_entropy(output))
                train_feature.append(feature.cpu())

        train_feature = [t.numpy() for t in train_feature] 
        np_entropy = np.array(train_entropy) 
        np.save('feature', train_feature)
        np.save('entropy', train_entropy)
        train_set = TensorDataset(torch.tensor(train_feature, dtype=torch.float), torch.tensor(train_entropy, dtype=torch.float))
        train_set = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
        self.train_set = train_set

    def training(self, epoch):
        train_loss = 0.0
        self.edm.train()
        tbar = tqdm(self.train_set)
        for i, (feature,entropy) in enumerate(tbar):
            if self.args.cuda:
                feature, entropy = feature.cuda(), entropy.cuda()
            output = self.edm(feature)
            loss = self.criterion(output, entropy)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d' % (epoch))
        print('Loss: %.3f' % train_loss)


def main():
    parser = argparse.ArgumentParser(description="Train EDM")

    """ model setting """
    parser.add_argument('--network', type=str, default='searched-dense', \
        choices=['searched-dense', 'autodeeplab-baseline', 'autodeeplab-dense'])
    parser.add_argument('--F', type=int, default=20)
    parser.add_argument('--B', type=int, default=5)
    parser.add_argument('--C', type=int, default=2, help='num of classifiers')


    """ dataset config"""
    parser.add_argument('--dataset', type=str, default='cityscapes', choices=['cityscapes', 'cityscapes_edm'], help='dataset name (default: pascal)')
    parser.add_argument('--workers', type=int, default=4, metavar='N', help='dataloader threads')
    

    """ training config """
    parser.add_argument('--epochs', type=int, default=10, metavar='N')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1, metavar='N')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N')
    parser.add_argument('--train-batch', type=int, default=16, metavar='N')
    parser.add_argument('--dist', action='store_true', default=False)


    """ optimizer params """
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR')
    parser.add_argument('--clean-module', type=int, default=0)
    parser.add_argument('--sync-bn', type=bool, default=False, help='whether to use sync bn (default: auto)')


    """ cuda, seed and logging """
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--gpu-ids', type=str, default='0', help='use which gpu to train, must be a comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S')


    """ checking point """
    parser.add_argument('--resume', type=str, default=None, help='put the path to resuming file if needed')
    parser.add_argument('--saved-arch-path', type=str, default='searched_arch/')
    parser.add_argument('--checkname', type=str, default='edm')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')


    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.network)

    print(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    new_trainer = trainNew(args)
    # new_trainer.mac()

    # new_trainer.make_data(args.train_batch)
    print('start training')
    for epoch in range(args.epochs):
        new_trainer.training(epoch)
    new_trainer.saver.save_checkpoint({
                'epoch':args.epochs,
                'state_dict': new_trainer.edm.state_dict(),
                'optimizer': new_trainer.optimizer.state_dict(),
                'best_pred': 1},
                True)
    new_trainer.writer.close()

if __name__ == "__main__":
   main()
