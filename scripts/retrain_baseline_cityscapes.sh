CUDA_VISIBLE_DEVICES=1 python ../train.py \
 --batch-size 12 --dataset cityscapes --checkname baseline_1m \
 --epoch 4200 --filter_multiplier 20 --backbone autodeeplab \
 --network autodeeplab --use-balanced-weights \
 --resize 1025 --crop_size 769 \
 --workers 12 --lr 0.05 --nesterov \
 --saved-arch-path ../searched_arch
