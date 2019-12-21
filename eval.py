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
from modeling.new_model import *
from torchviz import make_dot, make_dot_from_trace


class trainNew(object):
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

        cell_path_d = os.path.join(args.saved_arch_path, 'genotype_device.npy')
        cell_path_c = os.path.join(args.saved_arch_path, 'genotype_cloud.npy')
        network_path_space = os.path.join(args.saved_arch_path, 'network_path_space.npy')

        new_cell_arch_d = np.load(cell_path_d)
        new_cell_arch_c = np.load(cell_path_c)
 
        new_network_arch = np.load(network_path_space)
        
        if args.network == 'dist':
            new_network_arch = [0, 1, 2, 2, 3, 2, 2, 1, 2, 1, 1, 2]
            B_d=4

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
            new_cell_arch_d = cell
            new_cell_arch_c = cell      
            B_d=5

        # Define network
        model = new_cloud_Model(network_arch= new_network_arch,
                         cell_arch_d = new_cell_arch_d,
                         cell_arch_c = new_cell_arch_c,
                         num_classes=self.nclass,
                         device_num_layers=6,
                         B_d=B_d,
                         sync_bn=args.sync_bn)


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
        self.evaluator_device.reset()
        self.evaluator_cloud.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        time_meter_d = AverageMeter()
        time_meter_c = AverageMeter()

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            with torch.no_grad():
                device_output, cloud_output, time_1, time_2 = self.model(image, evaluation=True)

            device_loss = self.criterion(device_output, target)
            cloud_loss = self.criterion(cloud_output, target)
            loss = (device_loss + cloud_loss) /2
            test_loss += loss.item()

            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))

            pred_d = device_output.data.cpu().numpy()
            pred_c = cloud_output.data.cpu().numpy()

            target_show = target
            target = target.cpu().numpy()

            pred_d = np.argmax(pred_d, axis=1)
            pred_c = np.argmax(pred_c, axis=1)
            # Add batch sample into evaluator
            self.evaluator_device.add_batch(target, pred_d)
            self.evaluator_cloud.add_batch(target, pred_c)
            self.summary.visualize_image(self.writer, self.args.dataset, image, target_show, cloud_output, i)
            time_meter_d.update(time_1)
            time_meter_c.update(time_2)

        mIoU_d = self.evaluator_device.Mean_Intersection_over_Union()
        mIoU_c = self.evaluator_cloud.Mean_Intersection_over_Union()

        self.writer.add_scalar('val/device/mIoU', mIoU_d, 0)
        self.writer.add_scalar('val/cloud/mIoU', mIoU_c, 0)

        print('Validation:')
        print("device_mIoU:{}, cloud_mIoU: {}".format(mIoU_d, mIoU_c))
        print('Loss: %.3f' % test_loss)
        print("device_inference_time:{}, cloud_inference_time: {}".format(time_meter_d.average(), time_meter_c.average()))

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--network', type=str, default='dist',
                        choices=['dist', 'autodeeplab'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--dataset', type=str, default='cityscapes',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
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
    parser.add_argument('--test-batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')

    parser.add_argument('--clean-module', type=int, default=0)
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--use_amp', action='store_true', default=
                        False)  
    parser.add_argument('--seed', type=int, default=2, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--saved-arch-path', type=str, default=None,
                        help='put the path to alphas and betas')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # evaluation option
    parser.add_argument('--filter_multiplier', type=int, default=20)
    parser.add_argument('--autodeeplab', type=str, default='train',
                        choices=['search', 'train'])

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.checkname is None:
        args.checkname = 'evaluation'
    print(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    new_trainer = trainNew(args)

    new_trainer.validation(epoch)
    new_trainer.writer.close()

if __name__ == "__main__":
   main()

