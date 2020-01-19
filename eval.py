import argparse
import os
import numpy as np
from tqdm import tqdm
from mypath import Path
from dataloaders import make_data_loader
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from utils.eval_utils import AverageMeter
from modeling.dense_model import *
from torchviz import make_dot, make_dot_from_trace
import time

class Evaluation(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True, 'drop_last': True}
        _, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        cell_path_1 = os.path.join(args.saved_arch_path, 'genotype_1.npy')
        cell_path_2 = os.path.join(args.saved_arch_path, 'genotype_2.npy')
        network_path_space = os.path.join(args.saved_arch_path, 'network_path_space.npy')

        new_cell_arch_1 = np.load(cell_path_1)
        new_cell_arch_2 = np.load(cell_path_2)
 
        new_network_arch = np.load(network_path_space)
        
        if args.network == 'searched_arch':
            new_network_arch = [0, 1, 2, 2, 3, 2, 2, 1, 2, 1, 1, 2]

        elif args.network == 'autodeeplab':
            new_network_arch = [0, 0, 0, 1, 2, 1, 2, 2, 3, 3, 2, 1]
            cell = np.zeros((10, 2))
            cell[0] = [0, 7]
            cell[1] = [1, 4]
            cell[2] = [2, 4]
            cell[3] = [3, 6]
            cell[4] = [5, 4]
            cell[5] = [8, 4]
            cell[6] = [11, 5]
            cell[7] = [13, 5]
            cell[8] = [19, 7]
            cell[9] = [18, 5]
            cell=np.int_(cell)
            new_cell_arch_1 = cell
            new_cell_arch_2 = cell      

        # Define network
        model = Model_2(network_arch= new_network_arch,
                         cell_arch_d = new_cell_arch_1,
                         cell_arch_c = new_cell_arch_2,
                         num_classes=self.nclass,
                         device_num_layers=6,
                         sync_bn=args.sync_bn)

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
        self.evaluator_1 = Evaluator(self.nclass)
        self.evaluator_2 = Evaluator(self.nclass)

        # Using cuda
        if args.cuda:
            self.model = self.model.cuda()

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


            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))


    def validation(self):
        self.model.eval()
        self.evaluator_1.reset()
        self.evaluator_2.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        time_meter_1 = AverageMeter()
        time_meter_2 = AverageMeter()
        pred_1_meter = AverageMeter()
        pred_2_meter = AverageMeter()

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            with torch.no_grad():
                output_1, output_2, time_1, time_2 = self.model(image, evaluation=True)

            loss_1 = self.criterion(output_1, target)
            loss_2 = self.criterion(output_2, target)
            loss = (loss_1 + loss_2) /2
            test_loss += loss.item()

            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))

            torch.cuda.synchronize()
            tic = time.perf_counter()

            pred_1 = output_1.data.cpu().numpy()
            pred_1 = np.argmax(pred_1, axis=1)

            torch.cuda.synchronize()
            pred_1_meter.update(tic - time.perf_counter())

            torch.cuda.synchronize()
            tic = time.perf_counter()

            pred_2 = cloud_output.data.cpu().numpy()
            pred_2 = np.argmax(pred_2, axis=1)
            torch.cuda.synchronize()
            pred_2_meter.update(tic - time.perf_counter())

            target_show = target
            target = target.cpu().numpy()

            # Add batch sample into evaluator
            self.evaluator_1.add_batch(target, pred_1)
            self.evaluator_2.add_batch(target, pred_2)
            self.summary.visualize_image(self.writer, self.args.dataset, image, target_show, output_1, i)
            time_meter_1.update(time_1)
            time_meter_2.update(time_2)

        mIoU_1 = self.evaluator_1.Mean_Intersection_over_Union()
        mIoU_2 = self.evaluator_2.Mean_Intersection_over_Union()

        self.writer.add_scalar('val/1/mIoU', mIoU_1, 0)
        self.writer.add_scalar('val/2/mIoU', mIoU_2, 0)

        print('Validation:')
        print("device_mIoU:{}, cloud_mIoU: {}".format(mIoU_1, mIoU_2))
        print('Loss: %.3f' % test_loss)
        print("device_inference_time:{}, cloud_inference_time: {}".format(time_meter_1.average(), time_meter_2.average()))
        print("device_pred_time:{}, cloud_pred_time: {}".format(pred_1_meter.average(), pred_2_meter.average()))

def main():
    
    """ model setting """
    parser.add_argument('--network', type=str, default='searched_dense',
                        choices=['searched_dense', 'searched_baseline', 'autodeeplab'])
    parser.add_argument('--num_model_1_layers', type=int, default=6)
    parser.add_argument('--F_2', type=int, default=20)
    parser.add_argument('--F_1', type=int, default=20)
    parser.add_argument('--B_2', type=int, default=5)
    parser.add_argument('--B_1', type=int, default=5)


    """ dataset config"""
    parser.add_argument('--dataset', type=str, default='cityscapes',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--workers', type=int, default=1,
                        metavar='N', help='dataloader threads')


    """ training config """
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False)


    parser.add_argument('--clean-module', type=int, default=0)


    """ cuda, seed and logging """
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')


    """ checking point """
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--saved-arch-path', type=str, default=None,
                        help='put the path to alphas and betas')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')


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

    if args.checkname is None:
        args.checkname = 'evaluation'
    print(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    evaluation = Evaluation(args)

    evaluation.validation()
    evaluation.writer.close()

if __name__ == "__main__":
   main()

