class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return 'your path to pascal dataset'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return 'your path to sbd dataset'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return 'your path to cityscapes dataset'     # foler that contains leftImg8bit/
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
