mkdir data result para
cd data
mkdir train valid

CUDA_VISIBLE_DEVICES=1 python dkn_train.py
