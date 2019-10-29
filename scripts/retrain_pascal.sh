CUDA_VISIBLE_DEVICES=0 python train_new_model.py \
 --batch-size 50 --dataset pascal --checkname retrain_pascal \
 --epoch 5000 --filter_multiplier 20 --backbone autodeeplab \
 --resize 1024 --crop_size 513 \
 --workers 40 --lr 0.05 \
 --saved-arch-path /home/user/DistNAS-simple/run/cityscapes/beta/experiment_0
# --use_amp --opt_level O2 \
# --saved-arch-path /home/user/DistNAS-simple/run/cityscapes/beta/experiment_0
