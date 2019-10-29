import argparse
import os
import numpy as np
from tqdm import tqdm

from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from new_model import *
import time
import apex
#try:
#    from apex import amp
#    APEX_AVAILABLE = True
#except ModuleNotFoundError:
APEX_AVAILABLE = False

torch.backends.cudnn.benchmark = True

class trainNew(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        self.use_amp = True if (APEX_AVAILABLE and args.use_amp) else False
        self.opt_level = args.opt_level

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        cell_path_d = os.path.join(args.saved_arch_path, 'genotype_device.npy')
        cell_path_c = os.path.join(args.saved_arch_path, 'genotype_cloud.npy')
        network_path_space = os.path.join(args.saved_arch_path, 'network_path_space.npy')

        new_cell_arch_d = np.load(cell_path_d)
        new_cell_arch_c = np.load(cell_path_c)
        new_network_arch = np.load(network_path_space)
        new_network_arch = [1, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2]
        # Define network
        model = new_cloud_Model(network_arch= new_network_arch,
                         cell_arch_d = new_cell_arch_d,
                         cell_arch_c = new_cell_arch_c,
                         num_classes=self.nclass,
                         device_num_layers=6)

        # TODO: look into these
        # TODO: ALSO look into different param groups as done int deeplab below
#        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
#                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]
#
        train_params = [{'params': model.parameters(), 'lr': args.lr}]
        # Define Optimizer
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset + '_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer

        # Define Evaluator
        self.evaluator_device = Evaluator(self.nclass)
        self.evaluator_cloud = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                      args.epochs, len(self.train_loader)) #TODO: use min_lr ?

        # Using cuda
        # if self.use_amp and args.cuda:
        keep_batchnorm_fp32 = True if (self.opt_level == 'O2' or self.opt_level == 'O3') else None
        self.model = self.model.cuda()
        #     # fix for current pytorch version with opt_level 'O1'
        #     if self.opt_level == 'O1' and torch.__version__ < '1.3':
        #         for module in self.model.modules():
        #             if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        #                 # Hack to fix BN fprop without affine transformation
        #                 if module.weight is None:
        #                     module.weight = torch.nn.Parameter(
        #                         torch.ones(module.running_var.shape, dtype=module.running_var.dtype,
        #                                    device=module.running_var.device), requires_grad=False)
        #                 if module.bias is None:
        #                     module.bias = torch.nn.Parameter(
        #                         torch.zeros(module.running_var.shape, dtype=module.running_var.dtype,
        #                                     device=module.running_var.device), requires_grad=False)

        #     # print(keep_batchnorm_fp32)
        if self.use_amp:
            self.model, [self.optimizer] = amp.initialize(
                self.model, [self.optimizer], opt_level=self.opt_level,
                keep_batchnorm_fp32=keep_batchnorm_fp32, loss_scale="dynamic")

            # print('cuda finished')

        #self.model = self.model.cuda()
        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']

            # if the weights are wrapped in module object we have to clean it
            if args.clean_module:
                self.model.load_state_dict(checkpoint['state_dict'])
                state_dict = checkpoint['state_dict']
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove 'module.' of dataparallel
                    new_state_dict[name] = v
                self.model.load_state_dict(new_state_dict)

            else:
                if (torch.cuda.device_count() > 1 or args.load_parallel):
                    self.model.module.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint['state_dict'])

            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def validation(self, epoch):
        self.model.eval()
        self.evaluator_device.reset()
        self.evaluator_cloud.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        latency = 0.0
        
        for _, i in enumerate(tbar):
            sample = i
            break
        image, target = sample['image'], sample['label']
        image, target = image.cuda(), target.cuda()

        for _ in range(300):
            #image, target = sample['image'], sample['label']
            #if self.args.cuda:
            #    image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                start = time.time()
                device_output, cloud_output = self.model(image)
                end = time.time()
            # device_loss = self.criterion(device_output, target)
            # cloud_loss = self.criterion(cloud_output, target)
            # loss = device_loss + cloud_loss
            # test_loss += loss.item()
            # tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            # pred_d = device_output.data.cpu().numpy()
            # pred_c = cloud_output.data.cpu().numpy()
            # target = target.cpu().numpy()
            # pred_d = np.argmax(pred_d, axis=1)
            # pred_c = np.argmax(pred_c, axis=1)
            # Add batch sample into evaluator
            # self.evaluator_device.add_batch(target, pred_d)
            # self.evaluator_cloud.add_batch(target, pred_c)
            latency += end - start
        # mIoU_d = self.evaluator_device.Mean_Intersection_over_Union()
        # mIoU_c = self.evaluator_cloud.Mean_Intersection_over_Union()
        # self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        # self.writer.add_scalar('val/device/mIoU', mIoU_d, epoch)
        self.writer.add_scalar('latency', latency / 300, epoch)
#        print('Validation:')
#        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        # print("device_mIoU:{}, cloud_mIoU: {}".format(mIoU_d, mIoU_c))
        # print('Loss: %.3f' % test_loss)

        # new_pred = (mIoU_d + mIoU_c)/2
        # if new_pred > self.best_pred:
        #     is_best = True
        #     self.best_pred = new_pred
        #     self.saver.save_checkpoint({
        #         'epoch': epoch + 1,
        #         'state_dict': self.model.module.state_dict(),
        #         'optimizer': self.optimizer.state_dict(),
        #         'best_pred': self.best_pred,
        #     }, is_best)

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet', 'autodeeplab'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--opt_level', type=str, default='O0',
                        choices=['O0', 'O1', 'O2', 'O3'],
                        help='opt level for half percision training (default: O0)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='cityscapes',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=True,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--crop_size', type=int, default=320,
                        help='crop image size')
    parser.add_argument('--resize', type=int, default=512,
                        help='resize image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--use_amp', action='store_true', default=
                        False)  
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--saved-arch-path', type=str, default=None,
                        help='put the path to alphas and betas')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    parser.add_argument('--filter_multiplier', type=int, default=20)
    parser.add_argument('--autodeeplab', type=str, default='train',
                        choices=['search', 'train'])
    parser.add_argument('--load-parallel', type=int, default=0)
    parser.add_argument('--min_lr', type=float, default=0.001) #TODO: CHECK THAT THEY EVEN DO THIS FOR THE MODEL IN THE PAPER

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 4500,
            'pascal': 50,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size


    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)
    new_trainer = trainNew(args)
    print('Starting Epoch:', new_trainer.args.start_epoch)
    print('Total Epoches:', new_trainer.args.epochs)
    for epoch in range(new_trainer.args.start_epoch, new_trainer.args.epochs):
        new_trainer.validation(epoch)

    new_trainer.writer.close()

if __name__ == "__main__":
   main()
