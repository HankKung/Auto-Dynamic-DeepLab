CUDA_VISIBLE_DEVICES=0 python train_autodeeplab.py \
 --batch-size 8 --dataset cityscapes --checkname normal_beta \
 --alpha_epoch 25 --epoch 50 --filter_multiplier 8 \
 --resize 512 --crop_size 321 \
 --base_size 512 \
 --lr 0.07 --arch-lr 0.009 \
 --opt_level O2 --use_amp \
# --resume '/home/user/DistNAS-simple/run/cityscapes/dist/experiment_7/checkpoint.pth.tar'
