import argparse
import os
import time
import numpy as np

from mypath import Path
from dataloaders import make_data_loader

from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from utils.eval_utils import AverageMeter

from modeling.baseline_model import *
from modeling.dense_model import *
from modeling.operations import normalized_shannon_entropy
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.sync_batchnorm.replicate import patch_replication_callback

from tqdm import tqdm
from torchviz import make_dot, make_dot_from_trace
from apex import amp
from ptflops import get_model_complexity_info


class Evaluation(object):
    def __init__(self, args):

        self.args = args

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True, 'drop_last': True}
        _, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        if args.network == 'searched_dense':
            """ 40_5e_lr_38_31.91  """
            cell_path_1 = os.path.join(args.saved_arch_path, '40_5e_38_lr', 'genotype_1.npy')
            cell_path_2 = os.path.join(args.saved_arch_path, '40_5e_38_lr','genotype_2.npy')
            cell_arch_1 = np.load(cell_path_1)
            cell_arch_2 = np.load(cell_path_2)
            network_arch = [1, 2, 3, 2, 3, 2, 2, 1, 2, 1, 1, 2]
            low_level_layer = 0

            model = Model_2(network_arch,
                            cell_arch_1,
                            cell_arch_2,
                            self.nclass,
                            args,
                            low_level_layer)

        elif args.network == 'searched_baseline':
            cell_path_1 = os.path.join(args.saved_arch_path, 'searched_baseline', 'genotype_1.npy')
            cell_path_2 = os.path.join(args.saved_arch_path, 'searched_baseline','genotype_2.npy')
            cell_arch_1 = np.load(cell_path_1)
            cell_arch_2 = np.load(cell_path_2)
            network_arch = [0, 1, 2, 2, 3, 2, 2, 1, 2, 1, 1, 2]
            low_level_layer = 1
            model = Model_2_baseline(network_arch,
                                        cell_arch_1,
                                        cell_arch_2,
                                        self.nclass,
                                        args,
                                        low_level_layer)

        elif args.network.startswith('autodeeplab'):
            network_arch = [0, 0, 0, 1, 2, 1, 2, 2, 3, 3, 2, 1]
            cell_path = os.path.join(args.saved_arch_path, 'autodeeplab', 'genotype.npy')
            cell_arch = np.load(cell_path)
            low_level_layer = 2

            if args.network == 'autodeeplab-dense':
                model = Model_2(network_arch,
                                        cell_arch,
                                        cell_arch,
                                        self.nclass,
                                        args,
                                        low_level_layer)

            elif args.network == 'autodeeplab-baseline':
                model = Model_2_baseline(network_arch,
                                        cell_arch,
                                        cell_arch,
                                        self.nclass,
                                        args,
                                        low_level_layer)

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
        self.model = model

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


        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            with torch.no_grad():
                output_1, output_2 = self.model_forward_time(image)

            loss_1 = self.criterion(output_1, target)
            loss_2 = self.criterion(output_2, target)


            pred_1 = torch.argmax(output_1, axis=1)
            pred_2 = torch.argmax(output_2, axis=1)

            # Add batch sample into evaluator
            self.evaluator_1.add_batch(target, pred_1)
            self.evaluator_2.add_batch(target, pred_2)

        mIoU_1 = self.evaluator_1.Mean_Intersection_over_Union()
        mIoU_2 = self.evaluator_2.Mean_Intersection_over_Union()

        print('Validation:')
        print("mIoU_1:{}, mIoU_2: {}".format(mIoU_1, mIoU_2))


    def testing_entropy(self):
        self.saver = Saver(self.args)
        self.saver.save_experiment_config()

        """ Define Tensorboard Summary """
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

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
                output_1, avg_confidence, max_confidence = self.model.forward_testing_entropy(image)

            loss_1 = self.criterion(output_1, target)

            entropy = normalized_shannon_entropy(output_1)

            self.writer.add_scalar('avg_confidence/i', avg_confidence.item(), i)
            self.writer.add_scalar('max_confidence/i', max_confidence.item(), i)
            self.writer.add_scalar('entropy/i', entropy.item(), i)
            self.writer.add_scalar('loss/i', loss_1.item(), i)

            self.summary.visualize_image(self.writer, self.args.dataset, image, target_show, output_2, global_step)


        print('testing confidence')
        self.writer.close()


    def dynamic_inference(self, entropy=False, confidence_mode=False, pool_threshold=False, entropy_threshold=False):
        self.model.eval()
        self.evaluator_1.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            with torch.no_grad():
                output, earlier_exit, confidence = self.model.forward_dynamic_inference(image, \
                                                    entropy, confidence_mode, pool_threshold, entropy_threshold)

            loss = self.criterion(output, target)


            pred = torch.argmax(output, axis=1)

            # Add batch sample into evaluator
            self.evaluator_1.add_batch(target, pred)

        mIoU = self.evaluator_1.Mean_Intersection_over_Union()

        print('Validation:')
        print("mIoU_1:".format(mIoU_1))


    def mac(self):
        self.model.eval()
        with torch.no_grad():
            flops, params = get_model_complexity_info(self.model, (3, 1025, 2049), as_strings=True, print_per_layer_stat=False)
            print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
            print('{:<30}  {:<8}'.format('Number of parameters: ', params))


def main():
    parser = argparse.ArgumentParser(description="Eval")
    """ model setting """
    parser.add_argument('--network', type=str, default='searched_dense', choices=['searched_dense', 'searched_baseline', 'autodeeplab-baseline', 'autodeeplab-dense', 'supernet'])
    parser.add_argument('--num_model_1_layers', type=int, default=6)
    parser.add_argument('--F', type=int, default=20)
    parser.add_argument('--B', type=int, default=5)


    """ dynamic inference"""
    parser.add_argument('--entropy', type=bool, default=False)
    parser.add_argument('--confidence_mode', type=str, default='avg', choices=['avg', 'max'])
    parser.add_argument('--pool_threshold', type=float, default=None)
    parser.add_argument('--entropy_threshold', type=float, default=None)


    """ dataset config"""
    parser.add_argument('--dataset', type=str, default='cityscapes')
    parser.add_argument('--workers', type=int, default=1, metavar='N')


    """ training config """
    parser.add_argument('--use-amp', type=bool, default=False)
    parser.add_argument('--opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2', 'O3'])
    parser.add_argument('--sync-bn', type=bool, default=None)
    parser.add_argument('--freeze-bn', type=bool, default=False)

    parser.add_argument('--batch-size', type=int, default=1, metavar='N')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False)
    parser.add_argument('--clean-module', type=int, default=0)


    """ cuda, seed and logging """
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--gpu-ids', type=str, default='0')
    parser.add_argument('--seed', type=int, default=1, metavar='S')


    """ checking point """
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--saved-arch-path', type=str, default='searched_arch/')
    parser.add_argument('--checkname', type=str, default='testing')


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
    evaluation.testing_entropy()
    # evaluation.mac()
    #evaluation.validation()
    #evaluation.writer.close()

if __name__ == "__main__":
   main()

