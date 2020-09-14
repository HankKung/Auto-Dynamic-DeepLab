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
from modeling.autodeeplab import *
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
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True, 'drop_last': True}
        _, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        if args.network == 'searched-dense':
            """ 40_5e_lr_38_31.91  """
            cell_path = os.path.join(args.saved_arch_path, 'autodeeplab', 'genotype.npy')
            cell_arch = np.load(cell_path)
            network_arch = [1, 2, 2, 2, 3, 2, 2, 1, 1, 1, 1, 2]
            low_level_layer = 0

            model = Model_2(network_arch,
                            cell_arch,
                            self.nclass,
                            args,
                            low_level_layer)

        elif args.network == 'searched-baseline':
            cell_path = os.path.join(args.saved_arch_path, 'searched_baseline', 'genotype.npy')
            cell_arch = np.load(cell_path)
            network_arch = [0, 1, 2, 2, 3, 2, 2, 1, 2, 1, 1, 2]
            low_level_layer = 1
            model = Model_2_baseline(network_arch,
                                    cell_arch,
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
                                self.nclass,
                                args,
                                low_level_layer)

            elif args.network == 'autodeeplab-baseline':
                model = Model_2_baseline(network_arch,
                                        cell_arch,
                                        self.nclass,
                                        args,
                                        low_level_layer)
            elif args.network == 'autodeeplab':
                model = AutoDeepLab(network_arch,
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
        self.model.eval()
        self.evaluator_1.reset()
        self.evaluator_2.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        pool_vec = np.zeros(500)
        entropy_vec = np.zeros(500)
        loss_vec = np.zeros(500)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            with torch.no_grad():
                output_1, output_2, pool = self.model.dynamic_inference(image, threshold=threshold, confidence=confidence)

            loss_1 = self.criterion(output_1, target)
            loss_2 = self.criterion(output_2, target)


            pred_1 = torch.argmax(output_1, axis=1)
            pred_2 = torch.argmax(output_2, axis=1)

            entropy = normalized_shannon_entropy(output_1)

            # Add batch sample into evaluator
            self.evaluator_1.add_batch(target, pred_1)
            self.evaluator_2.add_batch(target, pred_2)

            self.writer.add_scalar('pool/i', pool.item(), i)
            self.writer.add_scalar('entropy/i', entropy, i)
            self.writer.add_scalar('loss/i', loss_1.item(), i)

            pool_vec[i] = pool.item()
            entropy_vec[i] = entropy
            loss_vec[i] = loss_1.item()

        pool_vec = torch.from_numpy(pool_vec)
        entropy_vec = torch.from_numpy(entropy_vec)
        loss_vec = torch.from_numpy(loss_vec)

        mIoU_1 = self.evaluator_1.Mean_Intersection_over_Union()
        mIoU_2 = self.evaluator_2.Mean_Intersection_over_Union()

        cos = nn.CosineSimilarity(dim=-1)
        cos_sim = cos(pool_vec, entropy_vec)
        print("pool-entropy_cosine similarity: {}".format(cos_sim))
        cos_sim = cos(pool_vec, loss_vec)
        print("pool-loss_cosine similarity: {}".format(cos_sim))
        cos_sim = cos(entropy_vec, loss_vec)
        print("-entropy-loss_cosine similarity: {}".format(cos_sim))

        print('Validation:')
        print("mIoU_1:{}, mIoU_2: {}".format(mIoU_1, mIoU_2))

    def dynamic_inference(self, threshold, confidence):
        self.model.eval()
        self.evaluator_1.reset()
        time_meter = AverageMeter()
        if confidence == 'edm':
            self.edm.eval()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        total_earlier_exit = 0
        confidence_value_avg = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            with torch.no_grad():
                output, earlier_exit, tic, confidence_value = \
                self.model.dynamic_inference(image, threshold=threshold, confidence=confidence, edm=self.edm)
            total_earlier_exit += earlier_exit
            confidence_value_avg += confidence_value
            time_meter.update(tic)
            
            loss = self.criterion(output, target)
            pred = torch.argmax(output, axis=1)

            # Add batch sample into evaluator
            self.evaluator_1.add_batch(target, pred)
            tbar.set_description('earlier_exit_num: %.1f' % (total_earlier_exit))
        mIoU = self.evaluator_1.Mean_Intersection_over_Union()

        print('Validation:')
        print("mIoU: {}".format(mIoU))
        print("mean_inference_time: {}".format(time_meter.average()))
        print("fps: {}".format(1.0/time_meter.average()))
        print("num_earlier_exit: {}".format(total_earlier_exit/500*100))
        print("avg_confidence: {}".format(confidence_value_avg/500))

    def time_measure(self):
        time_meter_1 = AverageMeter()
        time_meter_2 = AverageMeter()
        self.model.eval()
        self.evaluator_1.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            with torch.no_grad():
                _, _, t1, t2 = self.model.time_measure(image)
            if t1 != None:
                time_meter_1.update(t1)
            time_meter_2.update(t2)
        if t1 != None:
            print(time_meter_1.average())
        print(time_meter_2.average())



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
        choices=['searched-dense', 'searched-baseline', 'autodeeplab-baseline', 'autodeeplab-dense', 'autodeeplab', 'supernet'])
    parser.add_argument('--num_model_1_layers', type=int, default=6)
    parser.add_argument('--F', type=int, default=20)
    parser.add_argument('--B', type=int, default=5)
    parser.add_argument('--use-map', type=bool, default=False)


    """ dynamic inference"""
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--confidence', type=str, default='pool', choices=['edm', 'pool', 'entropy', 'max'])

    """ dataset config"""
    parser.add_argument('--dataset', type=str, default='cityscapes')
    parser.add_argument('--workers', type=int, default=1, metavar='N')


    """ training config """
    parser.add_argument('--use-amp', type=bool, default=False)
    parser.add_argument('--dist', action='store_true', default=False)

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
    # evaluation.mac()
    evaluation.dynamic_inference(threshold=args.threshold, confidence=args.confidence)
    #evaluation.validation()
    evaluation.writer.close()

if __name__ == "__main__":
   main()