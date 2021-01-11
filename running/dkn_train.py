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

epochs = 10
history_size = 50
batch_size = 100

# Paths
data_path = os.path.join("data", "mind-dkn")
train_file = os.path.join(data_path, "train_mind.txt")
valid_file = os.path.join(data_path, "valid_mind.txt")
test_file = os.path.join(data_path, "test_mind.txt")
user_history_file = os.path.join(data_path, "user_history.txt")
infer_embedding_file = os.path.join(data_path, "infer_embedding.txt")

train_path = os.path.join(data_path, "train")
valid_path = os.path.join(data_path, "valid")
test_path = os.path.join(data_path, "test")
# not have file then download
if not os.path.exists(train_path):
    train_zip, valid_zip, test_zip = download_mind(size='large', dest_path=data_path)
    train_path, valid_path, test_path = extract_mind(train_zip, valid_zip, test_zip, root_folder=data_path)
if not os.path.exists(train_file):
    train_session, train_history = read_clickhistory(train_path, "behaviors.tsv")
    get_train_input(train_session, train_file)
if not os.path.exists(valid_file):
    valid_session, valid_history = read_clickhistory(valid_path, "behaviors.tsv")
    get_valid_input(valid_session, valid_file)
if not os.path.exists(test_file):
    test_session, test_history = read_test_clickhistory(test_path, "behaviors.tsv")
    get_test_input(test_session, test_file)
