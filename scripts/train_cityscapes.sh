CUDA_VISIBLE_DEVICES=0 python ../train_autodeeplab.py \
 --batch-size 8 --workers 8 --dataset cityscapes --checkname search \
 --alpha_epoch 30 --epoch 60 --filter_multiplier 8 \
 --resize 512 --crop_size 321 \
 --base_size 512 \
 --lr 0.08 --min_lr 0.003 --arch-lr 0.015 \
 --opt_level O2 --use_amp 
