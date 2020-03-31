import argparse
import os
import numpy as np
from tqdm import tqdm
import sys
import torch
import torch.nn as nn
from collections import OrderedDict
from mypath import Path
from dataloaders import make_data_loader

from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from utils.copy_state_dict import copy_state_dict
from utils.eval_utils import *

from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.model_search import Model_search
from modeling.model_path_search import *
from modeling.model_baseline_path_search import *
from decoding.decoding_formulas import Decoder

import apex


try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False


print('working with pytorch version {}'.format(torch.__version__))
print('with cuda version {}'.format(torch.version.cuda))
print('cudnn enabled: {}'.format(torch.backends.cudnn.enabled))
print('cudnn version: {}'.format(torch.backends.cudnn.version()))

torch.backends.cudnn.benchmark = True

class Trainer(object):
    def __init__(self, args):
        self.args = args

        """ Define Saver """
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        """ Define Tensorboard Summary """
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        self.use_amp = True if (APEX_AVAILABLE and args.use_amp) else False
        self.opt_level = args.opt_level

        kwargs = {'num_workers': args.workers, 'pin_memory': True, 'drop_last':True, 'drop_last': True}

        self.train_loaderA, self.train_loaderB, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                """ if so, which trainloader to use? """
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None

        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=255).cuda()

        """ Define network """
        if self.args.network == 'supernet':
            model = Model_search(self.nclass, 12, self.args)
        elif self.args.network == 'path_dense_supernet':
            cell_path = os.path.join(args.saved_arch_path, 'autodeeplab', 'genotype.npy')
            cell_arch = np.load(cell_path)
            model = Model_layer_search(self.nclass, 12, self.args, alphas=cell_arch)

        elif self.args.network == 'path_baseline_supernet':
            cell_path = os.path.join(args.saved_arch_path, 'autodeeplab', 'genotype.npy')
            cell_arch = np.load(cell_path)
            model = Model_layer_search_baseline(self.nclass, 12, self.args, alphas=cell_arch)

        else:
            return


        optimizer = torch.optim.SGD(
                model.weight_parameters(),
                args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay
            )

        self.model, self.optimizer = model, optimizer

        self.architect_optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                                    lr=args.arch_lr, betas=(0.9, 0.999),
                                                    weight_decay=args.arch_weight_decay)

        """ Define Evaluator """
        self.evaluator_1 = Evaluator(self.nclass)
        self.evaluator_2 = Evaluator(self.nclass)

        """ Define lr scheduler """
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                      args.epochs, len(self.train_loaderA), min_lr=args.min_lr)
        """ Using cuda """
        if args.cuda:
            self.model = self.model.cuda()

        """ mixed precision """
        if self.use_amp and args.cuda:
            keep_batchnorm_fp32 = True if (self.opt_level == 'O2' or self.opt_level == 'O3') else None

            """ fix for current pytorch version with opt_level 'O1' """
            if self.opt_level == 'O1' and torch.__version__ < '1.3':
                for module in self.model.modules():
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
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
            self.model, [self.optimizer, self.architect_optimizer] = amp.initialize(
                self.model, [self.optimizer, self.architect_optimizer], opt_level=self.opt_level,
                keep_batchnorm_fp32=keep_batchnorm_fp32, loss_scale="dynamic")

            print('cuda finished')


        """ Using data parallel"""
        if args.cuda and len(self.args.gpu_ids) >1:
            if self.opt_level == 'O2' or self.opt_level == 'O3':
                print('currently cannot run with nn.DataParallel and optimization level', self.opt_level)
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            print('training on multiple-GPUs')

        """ Resuming checkpoint """
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
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


    def training(self, epoch):
        train_loss = 0.0
        search_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loaderA)
        num_img_tr = len(self.train_loaderA)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output_1, output_2 = self.model(image)
            loss_1 = self.criterion(output_1, target)
            loss_2 = self.criterion(output_2, target)
            loss = loss_1 + loss_2
            if self.use_amp:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            self.optimizer.step()
            
            if epoch >= self.args.alpha_epoch:
                search = iter(self.train_loaderB).next()
                image_search, target_search = search['image'], search['label']
                if self.args.cuda:
                    image_search, target_search = image_search.cuda(), target_search.cuda()

                self.architect_optimizer.zero_grad()
                output_search_1, output_search_2 = self.model(image_search)

                arch_loss_1 = self.criterion(output_search_1, target_search)
                arch_loss_2 = self.criterion(output_search_2, target_search)
                arch_loss = arch_loss_1 + arch_loss_2
                if self.use_amp:
                    with amp.scale_loss(arch_loss, self.architect_optimizer) as arch_scaled_loss:
                       arch_scaled_loss.backward()
                else:
                    arch_loss.backward()
                self.architect_optimizer.step()
                search_loss += arch_loss.item()
                
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f --Search loss: %.3f' \
                % (train_loss/(i+1), search_loss/(i+1)))

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        self.decoder_save(epoch, miou=None, evaluation=False)


    def validation(self, epoch):
        self.model.eval()
        self.evaluator_1.reset()
        self.evaluator_2.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output_1, output_2 = self.model(image)
            loss_1 = self.criterion(output_1, target)
            loss_2 = self.criterion(output_2, target)
            loss = loss_1 + loss_2
            test_loss += loss.item()

            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))

            output_1 = torch.argmax(output_1, axis=1)
            output_2 = torch.argmax(output_2, axis=1)

            """ Add batch sample into evaluator"""
            self.evaluator_1.add_batch(target, output_1)
            self.evaluator_2.add_batch(target, output_2)
        mIoU_1 = self.evaluator_1.Mean_Intersection_over_Union()
        mIoU_2 = self.evaluator_2.Mean_Intersection_over_Union()

        """ FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union() """
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/classifier_1/mIoU', mIoU_1, epoch)
        self.writer.add_scalar('val/classifier_2/mIoU', mIoU_2, epoch)

        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.test_batch_size + image.data.shape[0]))
        print('Loss: %.3f' % test_loss)
        new_pred = (mIoU_1 + mIoU_2)/2
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            if torch.cuda.device_count() > 1:
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': state_dict,
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

        """ decode the arch """
        self.decoder_save(epoch, miou=new_pred, evaluation=True)


    def decoder_save(self, epoch, miou=None, evaluation=False):

        
        num = str(epoch)
        if evaluation:
            num = num + '_eval'
        try:
            dir_name = os.path.join(self.saver.experiment_dir, num)
            os.makedirs(dir_name)
        except:
            print('folder path error')

        decoder = Decoder(None,
                          self.model.betas,
                          self.args.B)

        result_paths, result_paths_space = decoder.viterbi_decode()

        betas = self.model.betas.data.cpu().numpy()

        network_path_filename = os.path.join(dir_name,'network_path')
        beta_filename = os.path.join(dir_name, 'betas')

        np.save(network_path_filename, result_paths)
        np.save(beta_filename, betas)

        if miou != None:
            with open(os.path.join(dir_name, 'miou.txt'), 'w') as f:
                    f.write(str(miou))
        if evaluation:
            self.writer.add_text('network_path', str(result_paths), epoch+1000)
            self.writer.add_text('miou', str(miou), epoch+1000)
        else:
            self.writer.add_text('network_path', str(result_paths), epoch)



def main():
    parser = argparse.ArgumentParser(description="The Search")

    """ Search Network """
    parser.add_argument('--network', type=str, default='supernet', \
        choices=['supernet', 'path_dense_supernet', 'path_baseline_supernet'])
    parser.add_argument('--F', type=int, default=8)
    parser.add_argument('--B', type=int, default=5)


    """ Training Setting """
    parser.add_argument('--start-epoch', type=int, default=0, metavar='N', help='start epochs (default:0)')
    parser.add_argument('--epochs', type=int, default=40, metavar='N', help='number of epochs to train (default: auto)')
    parser.add_argument('--alpha-epoch', type=int, default=20, metavar='N', help='epoch to start training alphas')
    parser.add_argument('--sync-bn', type=bool, default=None, help='whether to use sync bn (default: auto)')
    parser.add_argument('--clean-module', type=int, default=0)


    """ Dataset Setting """
    parser.add_argument('--dataset', type=str, default='cityscapes', choices=['pascal', 'coco', 'cityscapes', 'kd'])
    parser.add_argument('--use-sbd', action='store_true', default=False, help='whether to use SBD dataset (default: True)')
    parser.add_argument('--load-parallel', type=int, default=0)
    parser.add_argument('--workers', type=int, default=2, metavar='N', help='dataloader threads')
    parser.add_argument('--batch-size', type=int, default=2, metavar='N')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False, help='whether to use balanced weights (default: False)')


    """ optimizer params """
    parser.add_argument('--lr', type=float, default=0.025, metavar='LR')
    parser.add_argument('--min-lr', type=float, default=0.001)
    parser.add_argument('--arch-lr', type=float, default=3e-3, metavar='LR', help='learning rate for alpha and beta in architect searching process')
    parser.add_argument('--lr-scheduler', type=str, default='cos',choices=['poly', 'step', 'cos'])
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--arch-weight-decay', type=float, default=1e-3, metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False, help='whether use nesterov (default: False)')
    parser.add_argument('--use-amp', type=bool, default=False) 
    parser.add_argument('--opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2', 'O3'], help='opt level for half percision training (default: O0)')


    """ cuda, seed and logging """
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training') 
    parser.add_argument('--gpu-ids', type=str, default='0', help='use which gpu to train, must be a comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')


    """ checking point """
    parser.add_argument('--resume', type=str, default=None, help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None, help='set the checkpoint name')
    parser.add_argument('--saved-arch-path', type=str, default='../searched_arch/')

    """ evaluation option """
    parser.add_argument('--eval-interval', type=int, default=1, help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False, help='skip validation during training')


    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.cuda and len(args.gpu_ids) > 1:
        args.sync_bn = True
    else:
        args.sync_bn = False

    if args.test_batch_size is None:
        args.test_batch_size = 1


    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(args.start_epoch, args.epochs):
        trainer.training(epoch)
        if epoch >= args.epochs - 5 and not targs.no_val \
        and epoch % args.eval_interval == (args.eval_interval - 1) or epoch == args.alpha_epoch+1:
            trainer.validation(epoch)
    trainer.writer.close()

if __name__ == "__main__":
   main()