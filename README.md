# AutoDynamicDeepLab
This repository is for the IROS 2021 paper [ADD: A Fine-grained Dynamic Inference Architecture for Semantic Image Segmentation](https://ieeexplore.ieee.org/abstract/document/9636650).

Dynamic-Auto-DeepLab performs three-stage training by firstly searching for the architecture. Second, train the model with the searched network architecture. Third, train earlier-exit-decision.

**Modifiy path to your Cityscapes in mypath.py**


## Neural Architecture Search

**We search for the architecture on Cityscapes**

```
cd scripts
bash search_cityscapes.sh
```

**The searched architecture and searching progress can be seen by:**
```
tensorboard --logdir path-to-your-exp
```
## Train model:
**One can choose network to train by modified .sh file. Note that we the batch size is #GPU/16 since we use torch.distributed**

```
bash train_dist.sh
```

## Train earlier-decision-maker (EDM) with the feature processed by the model we just trained:
```
bash train_edm.sh
```

## Evaluation on Cityscapes:
```
bash eval.sh
```

## Requirements

* Pytorch version 1.0+

* Python 3

* tensorboardX

* pycocotools

* tqdm

* apex

## Citation

## Acknowledgement
[Auto-DeepLab](https://github.com/NoamRosenberg/AutoML)

[pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception)

[DeepLabv3.pytorch](https://github.com/chenxi116/DeepLabv3.pytorch)

[Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch)

