CUDA_VISIBLE_DEVICES=1 python eval.py \
 --batch-size 32 --dataset cityscapes --checkname test_32 \
 --epoch 100 --filter_multiplier 20 --backbone autodeeplab \
 --resize 1024 --crop_size 769 \
 --workers 12 --lr 0.05 \
 --saved-arch-path /home/user/DistNAS-simple/run/cityscapes/beta/experiment_0 \
