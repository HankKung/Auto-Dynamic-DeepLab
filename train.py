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

from modeling.baseline_model import *
from modeling.ADD import *
from modeling.operations import normalized_shannon_entropy
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.sync_batchnorm.replicate import patch_replication_callback

from apex import amp
from ptflops import get_model_complexity_info
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

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
        self.use_amp = self.args.use_amp
        self.opt_level = self.args.opt_level

        if self.args.dist:
            torch.distributed.init_process_group(backend="nccl", init_method='env://')
            local_rank = self.args.local_rank
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            dist.barrier()

        """ Define Dataloader """
        kwargs = {'num_workers': args.workers, 'pin_memory': True, 'drop_last': True}
        self.train_loader, self.val_loader, _, self.nclass = make_data_loader(args, **kwargs)

        """ Define Criterion """
        """ whether to use class balanced weights """
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset + '_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=255).cuda()
        # self.criterion = DataParallelCriterion(self.criterion, device_ids=self.args.gpu_ids).cuda()
        if args.network == 'searched-dense':
            cell_path = os.path.join(args.saved_arch_path, 'autodeeplab', 'genotype.npy')
            cell_arch = np.load(cell_path)
            if self.args.C == 2:
                C_index = [5]
                #4_15_80e_40a_03-lr_5e-4wd_6e-4alr_1e-3awd 513x513 batch 4
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


        """ Define Optimizer """
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)


        self.model, self.optimizer = model, optimizer

        """ Define Evaluator """
        self.evaluator = []
        for num in range(self.args.C):
            self.evaluator.append(Evaluator(self.nclass))

        """ Define lr scheduler """
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                      args.epochs, len(self.train_loader))

        if args.cuda:
            self.model = self.model.cuda()

        """ mixed precision """
        if self.use_amp and args.cuda:
            keep_batchnorm_fp32 = True if (self.opt_level == 'O2' or self.opt_level == 'O3') else None

            """ fix for current pytorch version with opt_level 'O1' """
            if self.opt_level == 'O1' and torch.__version__ < '1.3':
                for module in self.model.modules():
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) or isinstance(module, SynchronizedBatchNorm2d):
                        """ Hack to fix BN fprop without affine transformation """
                        if module.weight is None:
                            module.weight = torch.nn.Parameter(
                                torch.ones(module.running_var.shape, dtype=module.running_var.dtype,
                                           device=module.running_var.device), requires_grad=False)
                        if module.bias is None:
                            module.bias = torch.nn.Parameter(
                                torch.zeros(module.running_var.shape, dtype=module.running_var.dtype,
                                            device=module.running_var.device), requires_grad=False)

            # print(keep_batchnorm_fp32)
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level=self.opt_level,
                keep_batchnorm_fp32=keep_batchnorm_fp32, loss_scale="dynamic")


        if args.cuda and len(self.args.gpu_ids) >1:
            if self.opt_level == 'O2' or self.opt_level == 'O3':
                print('currently cannot run with nn.DataParallel and optimization level', self.opt_level)
            
            if self.args.dist:
                self.model = DistributedDataParallel(self.model, 
                                                    device_ids=[self.args.local_rank],
                                                    output_device=self.args.local_rank).cuda()
            else:
                self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            # patch_replication_callback(self.model)
            print('training on multiple-GPUs')


        """ Resuming checkpoint """
        self.best_pred = 0.0
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

            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        """ Clear start epoch if fine-tuning """
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
    
            outputs = self.model(image)
            # loss = self.model(image, True, target)
            loss = []
            for classifier_i in range(self.args.C):
                loss.append(self.criterion(outputs[classifier_i], target))
            # loss = self.model.calculate_loss(image, target)
            loss = sum(loss)/(self.args.C)

            if self.use_amp:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            if i % 50 == 0:
                tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d' % (epoch))
        print('Loss: %.3f' % train_loss)


    def validation(self, epoch):
        self.model.eval()
        for e in self.evaluator:
            e.reset()

        confidence_meter = []
        for _ in range(self.args.C):
            confidence_meter.append(AverageMeter())

        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            with torch.no_grad():
                outputs = self.model(image)
            loss = []
            for classifier_i in range(self.args.C):
                loss.append(self.criterion(outputs[classifier_i], target))

            loss = sum(loss)/(self.args.C)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))

            target_show = target

            prediction = []
            """ Add batch sample into evaluator """
            for classifier_i in range(self.args.C):
                pred = torch.argmax(outputs[classifier_i], axis=1)
                prediction.append(pred)
                self.evaluator[classifier_i].add_batch(target, prediction[classifier_i])
                confidence = normalized_shannon_entropy(outputs[classifier_i])
                confidence_meter[classifier_i].update(confidence)

            if epoch//100 == i:
                global_step = epoch
                self.summary.visualize_image(self.writer, self.args.dataset, image, target_show, outputs[-1], global_step)

        mIoU = []
        mean_confidence = []
        for classifier_i, e in enumerate(self.evaluator):
            mIoU.append(e.Mean_Intersection_over_Union())
            self.writer.add_scalar('val/classifier_' + str(classifier_i) + '/mIoU', mIoU[classifier_i], epoch)
            mean_confidence.append(confidence_meter[classifier_i].average())
            self.writer.add_scalar('val/classifier_' + str(classifier_i) + '/confidence', mean_confidence[classifier_i], epoch)


        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.test_batch_size + image.data.shape[0]))
        if self.args.C == 2:
            print("classifier_1_mIoU:{}, classifier_2_mIoU: {}".format(mIoU[0], mIoU[1]))
            print("classifier_1_confidence:{}, classifier_2_confidence: {}".format(mean_confidence[0], mean_confidence[1]))
        elif self.args.C == 3:
            print("classifier_1_mIoU:{}, classifier_2_mIoU:{}, classifier_3_mIoU:{}".format(mIoU[0], mIoU[1], mIoU[2]))
            print("classifier_1_confidence:{}, classifier_2_confidence:{}, classifier_3_confidence:{}".format(mean_confidence[0], mean_confidence[1], mean_confidence[2]))
        elif self.args.C ==4:
            print("classifier_1_mIoU:{}, classifier_2_mIoU:{}, classifier_3_mIoU:{}, classifier_4_mIoU:{}".format(mIoU[0], mIoU[1], mIoU[2], mIoU[3]))
            print("classifier_1_confidence:{}, classifier_2_confidence:{}, classifier_3_confidence:{}, classifier_4_confidence:{}".format(mean_confidence[0], mean_confidence[1], mean_confidence[2], mean_confidence[3]))
        print('Loss: %.3f' % test_loss)

        new_pred = sum(mIoU)/self.args.C
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


    def mac(self):
        self.model.eval()
        with torch.no_grad():
            flops, params = get_model_complexity_info(self.model, (3, 1025, 2049), as_strings=True, print_per_layer_stat=False)
            print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
            print('{:<30}  {:<8}'.format('Number of parameters: ', params))


def main():
    parser = argparse.ArgumentParser(description="Dynamic DeepLab Training")

    """ model setting """
    parser.add_argument('--network', type=str, default='searched-dense', \
        choices=['searched-dense', 'autodeeplab-baseline', 'autodeeplab-dense', 'autodeeplab'])
    parser.add_argument('--F', type=int, default=20)
    parser.add_argument('--B', type=int, default=5)
    parser.add_argument('--C', type=int, default=3, help='num of classifiers')


    """ dataset config"""
    parser.add_argument('--dataset', type=str, default='cityscapes', choices=['pascal', 'coco', 'cityscapes'], help='dataset name (default: pascal)')
    parser.add_argument('--workers', type=int, default=4, metavar='N', help='dataloader threads')
    parser.add_argument('--dist', action='store_true', default=False)
    parser.add_argument("--local_rank", type=int)
    """ training config """
    parser.add_argument('--use-amp', type=bool, default=True)
    parser.add_argument('--opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2', 'O3'], help='opt level for half percision training (default: O0)')
    parser.add_argument('--sync-bn', type=bool, default=None, help='whether to use sync bn (default: auto)')
    parser.add_argument('--epochs', type=int, default=2400, metavar='N')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=4, metavar='N')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False)


    """ optimizer params """
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR')
    parser.add_argument('--min_lr', type=float, default=0)
    parser.add_argument('--lr-scheduler', type=str, default='poly', choices=['poly', 'step', 'cos'])
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M')
    parser.add_argument('--clean-module', type=int, default=0)
    parser.add_argument('--weight-decay', type=float, default=4e-5, metavar='M', help='w-decay (default: 4e-5)')
    parser.add_argument('--nesterov', action='store_true', default=False, help='whether use nesterov (default: False)')


    """ cuda, seed and logging """
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--gpu-ids', type=str, default='0,1', help='use which gpu to train, must be a comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S')


    """ checking point """
    parser.add_argument('--resume', type=str, default=None, help='put the path to resuming file if needed')
    parser.add_argument('--saved-arch-path', type=str, default='searched_arch/')
    parser.add_argument('--checkname', type=str, default='c3_autodeeplab-dense')


    """ finetuning pre-trained models """
    parser.add_argument('--ft', action='store_true', default=False, help='finetuning on a different dataset')


    """ evaluation option """
    parser.add_argument('--eval-interval', type=int, default=100, help='evaluuation interval (default: 1)')
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.cuda and len(args.gpu_ids) > 1 and args.network != 'autodeeplab-baseline':
        args.sync_bn = True
    else:
        args.sync_bn = False

    if args.test_batch_size is None:
        args.test_batch_size = 1

    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.network)

    print(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    new_trainer = trainNew(args)
    # new_trainer.mac()
    print('Starting Epoch:', new_trainer.args.start_epoch)
    print('Total Epoches:', new_trainer.args.epochs)
    
    for epoch in range(new_trainer.args.start_epoch, new_trainer.args.epochs):
        torch.cuda.empty_cache()
        new_trainer.training(epoch)
        if epoch % args.eval_interval == (args.eval_interval - 1) \
         or epoch > new_trainer.args.epochs - 5:
            new_trainer.validation(epoch)
    new_trainer.writer.close()

if __name__ == "__main__":
   main()
