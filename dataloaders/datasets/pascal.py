from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr

class VOCSegmentation(Dataset):
    """
    PascalVoc dataset
    """
    CLASSES = [
      'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
      'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
      'motorbike', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
      'tv/monitor'
    ]

    NUM_CLASSES = 21

    def __init__(self,
                 root,
                 train=True
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self.root = root
        self.train = train

        _voc_root = os.path.join(self.root, 'VOC2012')
        _list_dir = os.path.join(_voc_root, 'list')

        if self.train:
            _list_f = os.path.join(_list_dir, 'train_aug.txt')
        else:
            _list_f = os.path.join(_list_dir, 'val.txt')

        self.images = []
        self.masks = []
        with open(_list_f, 'r') as lines:
          for line in lines:
            _image = _voc_root + line.split()[0]
            _mask = _voc_root + line.split()[1]
            assert os.path.isfile(_image)
            assert os.path.isfile(_mask)
            self.images.append(_image)
            self.masks.append(_mask)

        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        # Display stats
        print('Number of images : {:d}'.format(len(self.images)))

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.masks[index])
        sample = {'image': _img, 'label': _target}

        if self.train:
            return self.transform_tr(sample)
        else:
            return self.transform_val(sample)

    def transform_tr(self, sample):
        transform = tr.tain_preprocess((513,513), self.mean, self.std)
        return transform(sample)

    def transform_val(self, sample):
        transform = tr.eval_preprocess((513,513), self.mean, self.std)
        return transform(sample)

    def __str__(self):
        return 'VOC2012(split=' + str(self.split) + ')'


if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    voc_train = VOCSegmentation(args, split='train')

    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='pascal')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)


