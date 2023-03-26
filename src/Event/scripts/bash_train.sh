export CUDA_VISIBLE_DEVICES=8

#python train.py --batch_size=8 \
#--n_epochs=80 \
#--data_class='ace'

python train.py --batch_size=6 \
--n_epochs=80 \
--data_class='wiki_src'

python train.py --batch_size=6 \
--n_epochs=80 \
--data_class='wiki_info'


