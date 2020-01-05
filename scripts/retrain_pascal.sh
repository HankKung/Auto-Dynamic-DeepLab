CUDA_VISIBLE_DEVICES=0 python ../train.py \
 --batch-size 16 --dataset pascal --checkname retrain_pascal \
 --epoch 800 --filter_multiplier 20 --backbone autodeeplab \
 --resize 512 --crop_size 513 --nesterov --nesterov \
 --workers 16 --lr 0.05 \
 --saved-arch-path ../searched_arch

