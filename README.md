# Dynamic-AutoDeepLab

Dynamic Auto-DeepLab perform two stage paradigm by firstly searching for the architecture, then deriving the searched architecture. Secondly, train the weights of model from scratch. 
## Dynamic Neural Architecture Search

**We conduct the architecture on Cityscapes**

```
cd scripts
bash train_cityscapes.sh
```

## Train weights from scratch
```
cd scripts
bash retrain_cityscapes
```
## Requirements

## Citation

## Acknowledgement
[Auto-DeepLab](https://github.com/NoamRosenberg/AutoML)

[pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception)

[DeepLabv3.pytorch](https://github.com/chenxi116/DeepLabv3.pytorch)

[Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch)


[SqueezeNAS](https://github.com/ashaw596/squeezenas)
