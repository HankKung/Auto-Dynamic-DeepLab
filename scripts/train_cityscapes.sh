CUDA_VISIBLE_DEVICES=0,1 python ../train_autodeeplab.py \
 --batch-size 8 --workers 8 --dataset cityscapes --checkname search \
 --alpha_epoch 30 --epoch 70 --filter_multiplier 8 \
 --resize 512 --crop_size 321 \
 --base_size 512 --sync-bn True --gpu-ids 0,1 \
 --lr 0.08 --min_lr 0.003 --arch-lr 0.015 --weight-decay 5e-4 \
 --opt_level O2 --use_amp 
