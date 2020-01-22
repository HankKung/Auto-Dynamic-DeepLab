class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/home/user/models/research/deeplab/datasets/pascal_voc_seg/VOCdevkit/VOC2012'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/home/user/models/research/deeplab/datasets/pascal_voc_seg/VOCdevkit/VOC2012'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '../dataset/cityscapes/'     # foler that contains leftImg8bit/
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
