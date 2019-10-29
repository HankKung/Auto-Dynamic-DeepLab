CUDA_VISIBLE_DEVICES=0 python train_new_model.py \
 --batch-size 12 --dataset cityscapes --checkname baseline \
 --epoch 2100 --filter_multiplier 20 --backbone autodeeplab \
 --network autodeeplab \
 --resize 1025 --crop_size 769 \
 --workers 12 --lr 0.05 --nesterov \
 --saved-arch-path /home/user/DistNAS-simple/run/cityscapes/normal_beta/experiment_0
                                                                                      
