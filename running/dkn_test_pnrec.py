import sys
sys.path.append("../")

import os
from tempfile import TemporaryDirectory
import logging
import papermill as pm
import tensorflow as tf

from reco_utils.dataset.download_utils import maybe_download
from reco_utils.dataset.mind import (download_mind, 
                                     extract_mind, 
                                     read_clickhistory,
                                     read_test_clickhistory,
                                     get_train_input, 
                                     get_valid_input,
                                     get_test_input,
                                     get_user_history,
                                     get_words_and_entities,
                                     generate_embeddings) 
from reco_utils.recommender.deeprec.deeprec_utils import prepare_hparams
from reco_utils.recommender.deeprec.models.dkn import DKN
from reco_utils.recommender.deeprec.io.dkn_iterator import DKNTextIterator

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
print(f"System version: {sys.version}")
print(f"Tensorflow version: {tf.__version__}")
# file dir
data_dir = 'data'
# logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt='%I:%M:%S')
handler.setFormatter(formatter)
logger.handlers = [handler]

epochs = 5
history_size = 50
batch_size = 100

# Paths
data_path = os.path.join("data", "mind-dkn")
train_file = os.path.join(data_path, "train_mind.txt")
valid_file = os.path.join(data_path, "valid_mind.txt")
test_file = os.path.join(data_path, "test_mind.txt")
user_history_file = os.path.join(data_path, "user_history.txt")
infer_embedding_file = os.path.join(data_path, "infer_embedding.txt")
news_feature_file = os.path.join(data_path, "doc_feature.txt")
word_embeddings_file = os.path.join(data_path, "word_embeddings_5w_100.npy")
entity_embeddings_file = os.path.join(data_path, "entity_embeddings_5w_100.npy")

train_path = os.path.join(data_path, "train")
valid_path = os.path.join(data_path, "valid")
test_path = os.path.join(data_path, "test")

yaml_file = maybe_download(url="https://recodatasets.blob.core.windows.net/deeprec/deeprec/dkn/dkn_MINDsmall.yaml", 
                           work_directory=data_path)
hparams = prepare_hparams(yaml_file,
                          news_feature_file=news_feature_file,
                          user_history_file=user_history_file,
                          wordEmb_file=word_embeddings_file,
                          entityEmb_file=entity_embeddings_file,
                          epochs=epochs,
                          history_size=history_size,
                          batch_size=batch_size)

hparams.save_model = True
hparams.show_step = 5000
hparams.MODEL_DIR = 'para'

model = DKN(hparams, DKNTextIterator)
model.load_model('./para/epoch_5')

# model.run_test(valid_file, 14085557, save_model=True, validate=True)
model.run_test(test_file, 10388965, save_model=False, validate=True)


    
