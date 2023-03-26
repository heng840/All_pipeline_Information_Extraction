export CUDA_VISIBLE_DEVICES=9
python main.py

python main.py \
--data_path='data/chinese.xml' \
--saved_path='data/chinese_uncertain_plain_gcn_joint_doc.pkl'