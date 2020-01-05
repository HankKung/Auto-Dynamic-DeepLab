CUDA_VISIBLE_DEVICES=0,1 python ../train.py \
 --batch-size 16 --dataset cityscapes --checkname proposed_retrain_large \
 --epoch 8100 --backbone autodeeplab  --F_c 48\
 --network dist --use-balanced-weights \
 --resize 1024 --crop_size 769 \
 --workers 16 --lr 0.05 --nesterov --sync-bn True --gpu-ids 0,1 \
 --saved-arch-path ../searched_arch
