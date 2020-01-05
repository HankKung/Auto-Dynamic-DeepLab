CUDA_VISIBLE_DEVICES=0 python ../train_autodeeplab.py \
 --batch-size 2 --workers 8 --dataset cityscapes --checkname search \
 --alpha_epoch 20 --epoch 40 \
 --sync-bn True \
 --lr 0.025 --min_lr 0.001 --arch-lr 0.003 --weight-decay 5e-4 \
