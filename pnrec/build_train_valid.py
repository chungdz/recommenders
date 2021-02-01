import os
import json
import pickle
import argparse
import re
import pandas as pd
from tqdm import tqdm
import numpy as np

data_path = 'data'

print("Loading behaviors info")
f_his_beh = os.path.join(data_path, "train/his_behaviors.tsv")
his_beh = pd.read_csv(f_his_beh, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])

user_dict = {}
user_idx = 0
for uid, imp in tqdm(his_beh[['uid', 'imp']].values, total=his_beh.shape[0], desc='history behavior'):

    if uid not in user_dict:
        user_dict[uid] = {"pos": [], "neg": [], "idx": user_idx}
        user_idx += 1
    
    imp_list = str(imp).split(' ')
    for impre in imp_list:
        arr = impre.split('-')
        curn = arr[0]
        label = int(arr[1])
        if label == 0:
            user_dict[uid]["neg"].append(curn)
        elif label == 1:
            user_dict[uid]["pos"].append(curn)
        else:
            raise Exception('label error!')

f_train_beh = os.path.join(data_path, "train/target_behaviors.tsv")
train_beh = pd.read_csv(f_train_beh, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])

new_train_beh = []
for imp_id, uid, imp_time, hist, imps in tqdm(train_beh.values, total=train_beh.shape[0], desc='train behavior'):
    if uid not in user_dict:
        continue

    new_row = [imp_id, uid, imp_time, ' '.join(user_dict[uid]['pos']), imps]
    new_train_beh.append(new_row)

new_train_df = pd.DataFrame(new_train_beh, columns=["id", "uid", "time", "hist", "imp"])
new_train_df.to_csv('./data/train/final_behaviors.tsv', sep='\t', index=None, header=None)

f_dev_beh = os.path.join(data_path, "valid/behaviors.tsv")
dev_beh = pd.read_csv(f_dev_beh, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])

new_dev_beh = []
for imp_id, uid, imp_time, hist, imps in tqdm(dev_beh.values, total=dev_beh.shape[0], desc='dev behavior'):
    if uid not in user_dict:
        continue
    
    new_row = [imp_id, uid, imp_time, ' '.join(user_dict[uid]['pos']), imps]
    new_dev_beh.append(new_row)

new_dev_df = pd.DataFrame(new_dev_beh, columns=["id", "uid", "time", "hist", "imp"])
new_dev_df.to_csv('./data/valid/final_behaviors.tsv', sep='\t', index=None, header=None)

    

