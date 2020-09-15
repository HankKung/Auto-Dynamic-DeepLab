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
from modeling.ADD import *
from modeling.autodeeplab import *
from modeling.operations import normalized_shannon_entropy
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.sync_batchnorm.replicate import patch_replication_callback

from tqdm import tqdm
from torchviz import make_dot, make_dot_from_trace
from ptflops import get_model_complexity_info


class Evaluation(object):
    def __init__(self, args):

        self.args = args
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True, 'drop_last': True}
        _, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

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
        self.evaluator = []
        for num in range(self.args.C):
            self.evaluator.append(Evaluator(self.nclass))

        # Using cuda
        if args.cuda:
            self.model = self.model.cuda()
        if args.confidence == 'edm':
            self.edm = EDM()
            self.edm = self.edm.cuda()
        else:
            self.edm = False

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
        if args.resume_edm is not None:
            if not os.path.isfile(args.resume_edm):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume_edm))
            checkpoint = torch.load(args.resume_edm)

            # if the weights are wrapped in module object we have to clean it
            if args.clean_module:
                self.edm.load_state_dict(checkpoint['state_dict'])
                state_dict = checkpoint['state_dict']
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove 'module.' of dataparallel
                    new_state_dict[name] = v
                self.edm.load_state_dict(new_state_dict)

            else:
                self.edm.load_state_dict(checkpoint['state_dict'])


    def validation(self):
        self.model.eval()
        for e in self.evaluator:
            e.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            with torch.no_grad():
                outputs = self.model(image)

            prediction = []
            """ Add batch sample into evaluator """
            for classifier_i in range(self.args.C):
                pred = torch.argmax(outputs[classifier_i], axis=1)
                prediction.append(pred)
                self.evaluator[classifier_i].add_batch(target, prediction[classifier_i])


            # Add batch sample into evaluator
        mIoU = []
        for classifier_i, e in enumerate(self.evaluator):
            mIoU.append(e.Mean_Intersection_over_Union())

        print("classifier_1_mIoU:{}, classifier_2_mIoU: {}".format(mIoU[0], mIoU[1]))

    def dynamic_inference(self, threshold, confidence):
        self.model.eval()
        self.evaluator[0].reset()
        if confidence == 'edm':
            self.edm.eval()
        time_meter = AverageMeter()

        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        total_earlier_exit = 0
        confidence_value_avg = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            with torch.no_grad():
                output, earlier_exit, tic, confidence_value = self.model.dynamic_inference(image, threshold=threshold, confidence=confidence, edm=self.edm)
            total_earlier_exit += earlier_exit
            confidence_value_avg += confidence_value
            time_meter.update(tic)
            
            loss = self.criterion(output, target)
            pred = torch.argmax(output, axis=1)

            # Add batch sample into evaluator
            self.evaluator[0].add_batch(target, pred)
            
        mIoU = self.evaluator[0].Mean_Intersection_over_Union()

        print('Validation:')
        print("mIoU: {}".format(mIoU))
        print("mean_inference_time: {}".format(time_meter.average()))
        print("fps: {}".format(1.0/time_meter.average()))
        print("num_earlier_exit: {}".format(total_earlier_exit/500*100))
        print("avg_confidence: {}".format(confidence_value_avg/500))


    def mac(self):
        self.model.eval()
        with torch.no_grad():
            flops, params = get_model_complexity_info(self.model, (3, 1025, 2049), as_strings=True, print_per_layer_stat=False)
            print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
            print('{:<30}  {:<8}'.format('Number of parameters: ', params))


def main():
    parser = argparse.ArgumentParser(description="Eval")

    """ model setting """
    parser.add_argument('--network', type=str, default='searched-dense', \
        choices=['searched-dense', 'autodeeplab-baseline', 'autodeeplab-dense', 'autodeeplab'])
    parser.add_argument('--F', type=int, default=20)
    parser.add_argument('--B', type=int, default=5)
    parser.add_argument('--C', type=int, default=2, help='num of classifiers')


    """ dynamic inference"""
    parser.add_argument('--dynamic', action='store_true', default=False)
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--confidence', type=str, default='pool', choices=['edm', 'entropy', 'max'])


    """ dataset config"""
    parser.add_argument('--dataset', type=str, default='cityscapes')
    parser.add_argument('--workers', type=int, default=1, metavar='N')


    """ training config """    
    parser.add_argument('--sync-bn', type=bool, default=None)
    parser.add_argument('--freeze-bn', type=bool, default=False)

    parser.add_argument('--batch-size', type=int, default=1, metavar='N')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False)
    parser.add_argument('--clean-module', type=int, default=0)
    parser.add_argument('--dist', action='store_true', default=False)

    """ cuda, seed and logging """
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--gpu-ids', type=str, default='0')
    parser.add_argument('--seed', type=int, default=1, metavar='S')


    """ checking point """
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--resume_edm', type=str, default=None)
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
    evaluation.mac()
    if args.dynamic:
        evaluation.dynamic_inference(threshold=args.threshold, confidence=args.confidence)
    else:
        evaluation.validation()

    evaluation.writer.close()

if __name__ == "__main__":
   main()

