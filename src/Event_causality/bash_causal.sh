export CUDA_VISIBLE_DEVICES=7
python train.py


python train.py \
 --use_scheduler \
 --saved_models='saved_models/wo_scheduler' \
 --output_dir='output_wo_scheduler.txt'