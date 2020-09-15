# Dynamic-AutoDeepLab

Dynamic-Auto-DeepLab performs three-stage training by firstly searching for the architecture. Second, train the model with the searched network architecture. 

**Modified path to your Cityscapes in mypath.py**


## Neural Architecture Search

**We search for the architecture on Cityscapes**

```
cd scripts
bash search_cityscapes.sh
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

## Citation

## Acknowledgement
[Auto-DeepLab](https://github.com/NoamRosenberg/AutoML)

[pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception)

[DeepLabv3.pytorch](https://github.com/chenxi116/DeepLabv3.pytorch)

[Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch)

