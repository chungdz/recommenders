import sys
sys.path.append("../")
import os
import numpy as np
import zipfile
from tqdm import tqdm
from tempfile import TemporaryDirectory
import tensorflow as tf
import pandas as pd
import json

from reco_utils.recommender.deeprec.deeprec_utils import download_deeprec_resources 
from reco_utils.recommender.newsrec.newsrec_utils import prepare_hparams
from reco_utils.recommender.newsrec.models.nrms import NRMSModel
from reco_utils.recommender.newsrec.io.mind_iterator import MINDIterator
from reco_utils.recommender.newsrec.newsrec_utils import get_mind_data_set
from reco_utils.recommender.deeprec.deeprec_utils import cal_metric

print("System version: {}".format(sys.version))
print("Tensorflow version: {}".format(tf.__version__))

model_path = 'para'
data_path = 'data'
epochs = 15
seed = 42
batch_size = 32

# Options: demo, small, large, large have test
MIND_type = 'large'

# tmpdir = TemporaryDirectory()
# data_path = tmpdir.name

train_news_file = os.path.join(data_path, 'train', r'news.tsv')
train_behaviors_file = os.path.join(data_path, 'train', r'behaviors.tsv')
valid_news_file = os.path.join(data_path, 'valid', r'news.tsv')
valid_behaviors_file = os.path.join(data_path, 'valid', r'behaviors.tsv')
test_news_file = os.path.join(data_path, 'test', r'news.tsv')
test_behaviors_file = os.path.join(data_path, 'test', r'behaviors.tsv')
wordEmb_file = os.path.join(data_path, "utils", "embedding.npy")
userDict_file = os.path.join(data_path, "utils", "uid2index.pkl")
wordDict_file = os.path.join(data_path, "utils", "word_dict.pkl")
yaml_file = os.path.join(data_path, "utils", r'nrms.yaml')

mind_url, mind_train_dataset, mind_dev_dataset, mind_test_dataset, mind_utils = get_mind_data_set(MIND_type)

if not os.path.exists(train_news_file):
    download_deeprec_resources(mind_url, os.path.join(data_path, 'train'), mind_train_dataset)   
if not os.path.exists(valid_news_file):
    download_deeprec_resources(mind_url, \
                               os.path.join(data_path, 'valid'), mind_dev_dataset)
if not os.path.exists(test_news_file):
    download_deeprec_resources(mind_url, \
                               os.path.join(data_path, 'test'), mind_test_dataset)
if not os.path.exists(yaml_file):
    download_deeprec_resources(r'https://recodatasets.blob.core.windows.net/newsrec/', \
                               os.path.join(data_path, 'utils'), mind_utils)


train = pd.read_csv('./data/train/behaviors.tsv', sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
train['date'] = train['time'].apply(lambda x: x.split()[0])
his_day = train[train['date'] != '11/14/2019']
target_day = train[train['date'] == '11/14/2019']

his_day.drop(columns=['date']).to_csv('./data/train/his_behaviors.tsv', sep="\t", encoding="utf-8", header=None, index=None)
print('his ', his_day.shape)
target_day.drop(columns=['date']).to_csv('./data/train/target_behaviors.tsv', sep="\t", encoding="utf-8", header=None, index=None)
print('target ', target_day.shape)

f_train_news = os.path.join(data_path, "train/news.tsv")
f_dev_news = os.path.join(data_path, "dev/news.tsv")

print("Loading training news")
train_news = pd.read_csv(f_train_news, sep="\t", encoding="utf-8",
                        names=["newsid", "cate", "subcate", "title", "abs", "url", "title_ents", "abs_ents"],
                        quoting=3)

print("Loading dev news")
dev_news = pd.read_csv(f_dev_news, sep="\t", encoding="utf-8",
                        names=["newsid", "cate", "subcate", "title", "abs", "url", "title_ents", "abs_ents"],
                        quoting=3)

all_news = pd.concat([train_news, dev_news], ignore_index=True)
all_news = all_news.drop_duplicates("newsid")
all_news.to_csv('./data/all_news.tsv', sep="\t", encoding="utf-8", header=None, index=None)
