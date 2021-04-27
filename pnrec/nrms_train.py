import sys
sys.path.append("../")
import os
import numpy as np
import zipfile
import argparse
from tqdm import tqdm
from tempfile import TemporaryDirectory
import tensorflow as tf

from reco_utils.recommender.deeprec.deeprec_utils import download_deeprec_resources 
from reco_utils.recommender.newsrec.newsrec_utils import prepare_hparams
from reco_utils.recommender.newsrec.models.nrms import NRMSModel
from reco_utils.recommender.newsrec.models.lstur import LSTURModel
from reco_utils.recommender.newsrec.models.naml import NAMLModel
from reco_utils.recommender.newsrec.models.npa import NPAModel
from reco_utils.recommender.newsrec.io.mind_iterator import MINDIterator
from reco_utils.recommender.newsrec.newsrec_utils import get_mind_data_set
from reco_utils.recommender.deeprec.deeprec_utils import cal_metric

print("System version: {}".format(sys.version))
print("Tensorflow version: {}".format(tf.__version__))

parser = argparse.ArgumentParser()
parser.add_argument("--root", default="data", type=str)
parser.add_argument("--model_name", default="nrms", type=str)
opt = parser.parse_args()

model_path = 'para'
data_path = opt.root
epochs = 10
seed = 42
batch_size = 128

# Options: demo, small, large, large have test
MIND_type = 'large'

# tmpdir = TemporaryDirectory()
# data_path = tmpdir.name

# train_news_file = os.path.join(data_path, 'train', r'news.tsv')
news_file = os.path.join(data_path, 'all_news.tsv')
train_behaviors_file = os.path.join(data_path, 'train', r'final_behaviors.tsv')
# valid_news_file = os.path.join(data_path, 'valid', r'news.tsv')
valid_behaviors_file = os.path.join(data_path, 'valid', r'final_behaviors.tsv')
# test_news_file = os.path.join(data_path, 'test', r'news.tsv')
# test_behaviors_file = os.path.join(data_path, 'test', r'behaviors.tsv')
wordEmb_file = os.path.join(data_path, "utils", "embedding.npy")
userDict_file = os.path.join(data_path, "utils", "uid2index.pkl")
wordDict_file = os.path.join(data_path, "utils", "word_dict.pkl")
subvertDict_file = os.path.join(data_path, "utils", "subvert_dict.pkl")
yaml_file = os.path.join(data_path, "utils", '{}.yaml'.format(opt.model_name))

hparams = prepare_hparams(yaml_file, 
                          wordEmb_file=wordEmb_file,
                          wordDict_file=wordDict_file, 
                          userDict_file=userDict_file,
                          subvertDict_file=subvertDict_file,
                          batch_size=batch_size,
                          epochs=epochs,
                          show_step=10)
print(hparams)

iterator = MINDIterator
if opt.model_name == 'nrms':
    model = NRMSModel(hparams, iterator, seed=seed)
elif opt.model_name == 'npa': # model can not save
    model = NPAModel(hparams, iterator, seed=seed)
elif opt.model_name == 'lstur':
    model = LSTURModel(hparams, iterator, seed=seed)
elif opt.model_name == 'naml':
    model = NAMLModel(hparams, iterator, seed=seed)

# print(model.run_slow_eval(news_file, valid_behaviors_file))

model.fit(news_file, train_behaviors_file, news_file, valid_behaviors_file)

# model_path = os.path.join(model_path, "model")
# os.makedirs(model_path, exist_ok=True)

# model.model.save_weights(os.path.join(model_path, "nrms_ckpt"))

# group_impr_indexes, group_labels, group_preds = model.run_slow_eval(test_news_file, test_behaviors_file)

# res = cal_metric(group_labels, group_preds, hparams.metrics)
