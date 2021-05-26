import os
import json
import pickle
import argparse
import re
import pandas as pd
from tqdm import tqdm
import numpy as np

def build_word_embeddings(vocab, pretrained_embedding, weights_output_file):
    # Load 预训练的embedding
    lines = open(pretrained_embedding, "r", encoding="utf8").readlines()
    emb_dict = dict()
    error_line = 0
    embed_size = 0
    for line in lines:
        row = line.strip().split()
        try:
            embedding = [float(w) for w in row[1:]]
            emb_dict[row[0]] = np.array(embedding)
            if embed_size == 0:
                embed_size = len(embedding)
        except:
            error_line += 1
    print("Error lines: {}".format(error_line))

    weights_matrix = np.zeros((len(vocab), embed_size))
    words_found = 0

    for k, v in vocab.items():
        try:
            weights_matrix[v] = emb_dict[k]
            words_found += 1
        except KeyError:
            weights_matrix[v] = np.random.normal(size=(embed_size,))
    print("Totally find {} words in pre-trained embeddings.".format(words_found))
    np.save(weights_output_file, weights_matrix)
    print(weights_matrix.shape)

def parse_ent_list(x):
    if x.strip() == "":
        return ''

    return ' '.join([k["WikidataId"] for k in json.loads(x)])

punctuation = '!,;:?"\''
def removePunctuation(text):
    text = re.sub(r'[{}]+'.format(punctuation),'',text)
    return text.strip().lower()

parser = argparse.ArgumentParser()

parser.add_argument("--title_len", default=15, type=int, help="Max length of the title.")
parser.add_argument("--pos_hist_length", default=30, type=int)
parser.add_argument("--neg_hist_length", default=60, type=int)

args = parser.parse_args()

data_path = 'data'
max_title_len = args.title_len

print("Loading news info")
print("Loading training news")
all_news = pd.read_csv(os.path.join(data_path, "all_news.tsv"), sep="\t", encoding="utf-8",
                        names=["newsid", "cate", "subcate", "title", "abs", "url", "title_ents", "abs_ents"],
                        quoting=3)

word_dict = {'<pad>': 0}
word_idx = 1

for n, title, cate, subcate in tqdm(all_news[['newsid', "title", "cate", "subcate"]].values, total=all_news.shape[0], desc='parse news'):

    tarr = removePunctuation(title).split()
    
    for t in tarr:
        if t not in word_dict:
            word_dict[t] = word_idx
            word_idx += 1

print('all word', len(word_dict))

print("Loading behaviors info")
train_beh = pd.read_csv(os.path.join(data_path, "train/final_behaviors.tsv"), sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
dev_beh = pd.read_csv(os.path.join(data_path, "valid/final_behaviors.tsv"), sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
all_beh = pd.concat([train_beh, dev_beh], ignore_index=True)

user_dict = {}
user_idx = 0
for uid, imp in tqdm(all_beh[['uid', 'imp']].values, total=all_beh.shape[0], desc='history behavior'):

    if uid not in user_dict:
        user_dict[uid] = user_idx
        user_idx += 1
        
print('all user', len(user_dict))



build_word_embeddings(word_dict, 'data/glove.840B.300d.txt', 'data/utils/embedding.npy')
pickle.dump(user_dict, open('data/utils/uid2index.pkl', 'wb'))
pickle.dump(word_dict, open('data/utils/word_dict.pkl', 'wb'))

