# Distributed-AutoDeepLab

Distributed Auto-DeepLab perform two stage paradigm by firstly searching for the architecture, then deriving the searched architecture. Secondly, train the weights of model from scratch. 
## Distributed Neural Architecture Search
```
cd scripts
bash train_cityscapes.sh
```

## Train weights from scratch
```
cd scripts
bash retrain_cityscapes
```
