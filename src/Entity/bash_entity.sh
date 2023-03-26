export CUDA_VISIBLE_DEVICES=9
python run_entity.py \
    --learning_rate=1e-5 \
    --task_learning_rate=5e-4 \
    --train_batch_size=16 \
    --context_window=300 \
    --task=scierc\
    --data_dir='/home/chenyuheng/chenyuheng/DataSets/sciERC_processed/processed_data/json' \
    --model=bert-base-uncased \
    --output_dir='entity_output'
#    --do_train \
#    --do_eval \
#    --eval_test \
