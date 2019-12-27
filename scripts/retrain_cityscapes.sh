CUDA_VISIBLE_DEVICES=1 python ../train_new_model.py \
 --batch-size 16 --dataset cityscapes --checkname proposed_retrain \
 --epoch 2700 --backbone autodeeplab \
 --network dist --use-balanced-weights \
 --resize 1024 --crop_size 769 \
 --workers 16 --lr 0.05 --nesterov --gpu-ids 0,1\
 --saved-arch-path ../searched_arch
