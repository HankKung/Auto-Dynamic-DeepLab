from dataloaders.datasets import cityscapes, pascal
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def make_data_loader(args, **kwargs):

    if args.dataset == 'pascal':
        train_set = pascal.VOCSegmentation('../../../Pascal/VOCdevkit', train=True)
        val_set = pascal.VOCSegmentation('../../../Pascal/VOCdevkit', train=False)

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class

    elif 'cityscapes' in args.dataset:
        if args.dataset == 'cityscapes_edm':
            train_set = cityscapes.CityscapesSegmentation(args, split='train', full=True)
            num_class = train_set.NUM_CLASSES
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        else:
            if 'supernet' in args.network:
                train_set1, train_set2 = cityscapes.twoTrainSeg(args)
                num_class = train_set1.NUM_CLASSES
                train_loader1 = DataLoader(train_set1, batch_size=args.batch_size, shuffle=True, **kwargs)
                train_loader2 = DataLoader(train_set2, batch_size=args.batch_size, shuffle=True, **kwargs)
            else:
                train_set = cityscapes.CityscapesSegmentation(args, split='train')
                num_class = train_set.NUM_CLASSES
                if args.dist:
                    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, sampler=DistributedSampler(train_set), **kwargs)
                else:
                    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)

        if args.network != None:
            val_set = cityscapes.CityscapesSegmentation(args, split='val')
            test_set = cityscapes.CityscapesSegmentation(args, split='test')
        else:
            raise Exception('autodeeplab param not set properly')

        val_loader = DataLoader(val_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)


        if 'supernet' in args.network:
            return train_loader1, train_loader2, val_loader, test_loader, num_class
        else:
            return train_loader, val_loader, test_loader, num_class


    elif args.dataset == 'coco':
        train_set = coco.COCOSegmentation(args, split='train')
        val_set = coco.COCOSegmentation(args, split='val')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError

