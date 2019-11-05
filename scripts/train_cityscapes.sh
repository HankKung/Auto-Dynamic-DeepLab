CUDA_VISIBLE_DEVICES=1 python train_autodeeplab.py \
 --batch-size 8 --dataset cityscapes --checkname test_search \
 --alpha_epoch 25 --epoch 60 --filter_multiplier 8 \
 --resize 512 --crop_size 321 \
 --base_size 512 \
 --lr 0.08 --min_lr 0.003 --arch-lr 0.009 \
 --opt_level O2 --use_amp 
