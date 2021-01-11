mkdir data result para

CUDA_VISIBLE_DEVICES=1 python dkn_train.py
CUDA_VISIBLE_DEVICES=1 python dkn_test.py
