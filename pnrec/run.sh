mkdir data para
python data_prepocess.py
python build_train_valid.py
CUDA_VISIBLE_DEVICES=1 python nrms_train.py --model_name=lstur
