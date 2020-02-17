import torch
import random
import numpy as np
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageFilter
import math
class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'label': mask}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img,
                'label': mask}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}

# resize to 512*1024
class FixedResize(object):
    """change the short edge length to size"""
    def __init__(self, resize=512):
        self.resize = resize  # size= 512
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
#        print(img.size)

 #       print(mask.size)
        assert img.size == mask.size

        w, h = img.size
        pad_tb = max(0, self.resize[0] - h)
        pad_lr = max(0, self.resize[1] - w)  
        data_transforms = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
          ])
        img = data_transforms(img)
        mask = torch.LongTensor(np.array(mask).astype(np.int64)) 

        img = torch.nn.ZeroPad2d((0, pad_lr, 0, pad_tb))(img)
        mask = torch.nn.ConstantPad2d((0, pad_lr, 0, pad_tb), 255)(mask)      
        
        h, w = img.shape[1], img.shape[2]
        i = random.randint(0, h - self.resize[0])
        j = random.randint(0, w - self.resize[1])
        img = img[:, i:i + self.resize[0], j:j + self.resize[1]]
        mask = mask[i:i + self.resize[0], j:j + self.resize[1]]

        return {'image': img,
                'label': mask}

 # random crop 321*321
class RandomCrop(object):
    def __init__(self,  crop_size=769):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        return {'image': img,
                'label': mask}

class FixedResize_Search(object):
    """change the short edge length to size"""

    def __init__(self, resize=512):
        self.size1 = resize  # size= 512

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size

        w, h = img.size
        if w > h:
            oh = self.size1
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.size1
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        return {'image': img,
                'label': mask}

class Crop_for_eval(object):
    def __init__(self):
        self.fill=255

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = ImageOps.expand(img, border=(0, 0, 1, 1), fill=0)
        mask = ImageOps.expand(mask, border=(0, 0, 1, 1), fill=self.fill)

        return {'image': img,
                'label': mask}

class train_preprocess(object):
    def __init__(self, crop_size, mean, std, scale=0):
        self.crop_size = crop_size
        self.mean = mean
        self.std = std
        self.scale = scale

    def __call__(self, sample):
        image = sample['image']
        mask = sample['label']

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if self.scale == 0:
            scale=(0.5, 2.0)
            w, h = image.size
            rand_log_scale = math.log(scale[0], 2) + random.random() * (math.log(scale[1], 2) - math.log(scale[0], 2))
            random_scale = math.pow(2, rand_log_scale)
            new_size = (int(round(w * random_scale)), int(round(h * random_scale)))
            image = image.resize(new_size, Image.ANTIALIAS)
            mask = mask.resize(new_size, Image.NEAREST)
        else:
            w, h = image.size
            new_size = (int(round(w * self.scale)), int(round(h * self.scale)))
            image = image.resize(new_size, Image.ANTIALIAS)
            mask = mask.resize(new_size, Image.NEAREST)

        data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
            ])
        image = data_transforms(image)
        mask = torch.LongTensor(np.array(mask).astype(np.int64))

        h, w = image.shape[1], image.shape[2]
        pad_tb = max(0, self.crop_size[0] - h)
        pad_lr = max(0, self.crop_size[1] - w)
        image = torch.nn.ZeroPad2d((0, pad_lr, 0, pad_tb))(image)
        mask = torch.nn.ConstantPad2d((0, pad_lr, 0, pad_tb), 255)(mask)

        h, w = image.shape[1], image.shape[2]
        i = random.randint(0, h - self.crop_size[0])
        j = random.randint(0, w - self.crop_size[1])
        image = image[:, i:i + self.crop_size[0], j:j + self.crop_size[1]]
        mask = mask[i:i + self.crop_size[0], j:j + self.crop_size[1]]

        return {'image': image,
                'label': mask}


class eval_preprocess(object):
    def __init__(self, crop_size, mean, std):
        self.crop_size = crop_size
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image = sample['image']
        mask = sample['label']

        data_transforms = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize(self.mean, self.std)
        ])

        image = data_transforms(image)
        mask = torch.LongTensor(np.array(mask).astype(np.int64))

        h, w = image.shape[1], image.shape[2]
        pad_tb = max(0, self.crop_size[0] - h)
        pad_lr = max(0, self.crop_size[1] - w)
        image = torch.nn.ZeroPad2d((0, pad_lr, 0, pad_tb))(image)
        mask = torch.nn.ConstantPad2d((0, pad_lr, 0, pad_tb), 255)(mask)

        h, w = image.shape[1], image.shape[2]
        i = random.randint(0, h - self.crop_size[0])
        j = random.randint(0, w - self.crop_size[1])
        image = image[:, i:i + self.crop_size[0], j:j + self.crop_size[1]]
        mask = mask[i:i + self.crop_size[0], j:j + self.crop_size[1]]

        return {'image': image,
                'label': mask}

class full_image_eval_preprocess(object):
    def __init__(self, crop_size, mean, std):
        self.crop_size = crop_size
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image = sample['image']
        mask = sample['label']

        data_transforms = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize(self.mean, self.std)
        ])

        image = data_transforms(image)
        mask = torch.LongTensor(np.array(mask).astype(np.int64))

        h, w = image.shape[1], image.shape[2]
        pad_tb = max(0, self.crop_size[0] - h)
        pad_lr = max(0, self.crop_size[1] - w)
        image = torch.nn.ZeroPad2d((0, pad_lr, 0, pad_tb))(image)
        mask = torch.nn.ConstantPad2d((0, pad_lr, 0, pad_tb), 255)(mask)

        return {'image': image,
                'label': mask}

