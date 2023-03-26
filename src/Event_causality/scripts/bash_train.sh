export CUDA_VISIBLE_DEVICES=8
python train.py --batch_size_train=75 \
 --batch_size_test=20 \
 --epoch_nums=100 \
 --saved_models=saved_models/with_scheduler

#python train.py --use_scheduler \
# --batch_size_train=75 \
# --batch_size_test=20 \
# --epoch_nums=100 \
# --saved_models=saved_models/wo_scheduler