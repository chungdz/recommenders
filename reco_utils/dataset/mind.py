# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import random
import logging
import json
import numpy as np
from nltk.tokenize import RegexpTokenizer

from reco_utils.dataset.download_utils import maybe_download, download_path, unzip_file


URL_MIND_LARGE_TRAIN = (
    "https://mind201910small.blob.core.windows.net/release/MINDlarge_train.zip"
)
URL_MIND_LARGE_VALID = (
    "https://mind201910small.blob.core.windows.net/release/MINDlarge_dev.zip"
)
URL_MIND_LARGE_TEST = (
    "https://mind201910small.blob.core.windows.net/release/MINDlarge_test.zip"
)
URL_MIND_SMALL_TRAIN = (
    "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip"
)
URL_MIND_SMALL_VALID = (
    "https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip"
)
URL_MIND_DEMO_TRAIN = (
    "https://recodatasets.blob.core.windows.net/newsrec/MINDdemo_train.zip"
)
URL_MIND_DEMO_VALID = (
      "https://recodatasets.blob.core.windows.net/newsrec/MINDdemo_dev.zip"
)
URL_MIND_DEMO_UTILS = (
      "https://recodatasets.blob.core.windows.net/newsrec/MINDdemo_utils.zip"
)

URL_MIND = {
    "large": (URL_MIND_LARGE_TRAIN, URL_MIND_LARGE_VALID, URL_MIND_LARGE_TEST),
    "small": (URL_MIND_SMALL_TRAIN, URL_MIND_SMALL_VALID),
    "demo": (URL_MIND_DEMO_TRAIN, URL_MIND_DEMO_VALID)
}

logger = logging.getLogger()

def download_mind(size="small", dest_path=None):
    """Download MIND dataset

    Args:
        size (str): Dataset size. One of ["small", "large"]
        dest_path (str): Download path. If path is None, it will download the dataset on a temporal path
        
    Returns:
        str, str: Path to train and validation sets.
    """
    size_options = ["small", "large", "demo"]
    if size not in size_options:
        raise ValueError(f"Wrong size option, available options are {size_options}")
    if size == 'large':
        url_train, url_valid, url_test = URL_MIND[size]
        with download_path(dest_path) as path:
            train_path = maybe_download(url=url_train, work_directory=path)
            valid_path = maybe_download(url=url_valid, work_directory=path)
            test_path = maybe_download(url=url_test, work_directory=path)
        return train_path, valid_path, test_path
    else:
        url_train, url_valid = URL_MIND[size]
        with download_path(dest_path) as path:
            train_path = maybe_download(url=url_train, work_directory=path)
            valid_path = maybe_download(url=url_valid, work_directory=path)
        return train_path, valid_path


def extract_mind(train_zip, valid_zip, test_zip=None, train_folder="train", valid_folder="valid", test_folder='test', root_folder=""):
    """Extract MIND dataset

    Args:
        train_zip (str): Path to train zip file
        valid_zip (str): Path to valid zip file
        train_folder (str): Destination forder for train set
        valid_folder (str): Destination forder for validation set
    
    Returns:
        str, str: Train and validation folders
    """
    train_path = os.path.join(root_folder, train_folder)
    valid_path = os.path.join(root_folder, valid_folder)
    unzip_file(train_zip, train_path)
    unzip_file(valid_zip, valid_path)
    if test_zip is not None:
        test_path = os.path.join(root_folder, test_folder)
        unzip_file(test_zip, test_path)
        return train_path, valid_path, test_path
    return train_path, valid_path


def read_clickhistory(path, filename):
    """Read click history file

    Args:
        path (str): Folder path
        filename (str): Filename

    Returns:
        list, dict: List of user session with user_id, clicks, positive and negative interactions. Dictionary
            with user_id click history
    """
    userid_history = {}
    with open(os.path.join(path, filename)) as f:
        lines = f.readlines()
    sessions = []
    for i in range(len(lines)):
        _, userid, imp_time, click, imps = lines[i].strip().split("\t")
        clicks = click.split(" ")
        pos = []
        neg = []
        imps = imps.split(" ")
        for imp in imps:
            if imp.split("-")[1] == "1":
                pos.append(imp.split("-")[0])
            else:
                neg.append(imp.split("-")[0])
        userid_history[userid] = clicks
        sessions.append([userid, clicks, pos, neg])
    return sessions, userid_history

def read_test_clickhistory(path, filename):
    """Read click history file

    Args:
        path (str): Folder path
        filename (str): Filename

    Returns:
        list, dict: List of user session with user_id, clicks, all interactions. Dictionary
            with user_id click history
    """
    userid_history = {}
    with open(os.path.join(path, filename)) as f:
        lines = f.readlines()
    sessions = []
    for i in range(len(lines)):
        _, userid, imp_time, click, imps = lines[i].strip().split("\t")
        clicks = click.split(" ")
        imps = imps.split(" ")
        userid_history[userid] = clicks
        sessions.append([userid, clicks, imps])
    return sessions, userid_history

def _newsample(nnn, ratio):
    if ratio > len(nnn):
        return random.sample(nnn * (ratio // len(nnn) + 1), ratio)
    else:
        return random.sample(nnn, ratio)


def get_train_input(session, train_file_path, npratio=4):
    """Generate train file.

    Args:
        session (list): List of user session with user_id, clicks, positive and negative interactions.
        train_file_path (str): Path to file.
        npration (int): Ratio for negative sampling.
    """
    fp_train = open(train_file_path, "w", encoding="utf-8")
    for sess_id in range(len(session)):
        sess = session[sess_id]
        userid, _, poss, negs = sess
        for i in range(len(poss)):
            pos = poss[i]
            neg = _newsample(negs, npratio)
            fp_train.write("1 " + "train_" + userid + " " + pos + "\n")
            for neg_ins in neg:
                fp_train.write("0 " + "train_" + userid + " " + neg_ins + "\n")
    fp_train.close()
    if os.path.isfile(train_file_path):
        logger.info(f"Train file {train_file_path} successfully generated")
    else:
        raise FileNotFoundError(f"Error when generating {train_file_path}")


def get_valid_input(session, valid_file_path):
    """Generate validation file.

    Args:
        session (list): List of user session with user_id, clicks, positive and negative interactions.
        valid_file_path (str): Path to file.
    """
    fp_valid = open(valid_file_path, "w", encoding="utf-8")
    for sess_id in range(len(session)):
        userid, _, poss, negs = session[sess_id]
        for i in range(len(poss)):
            fp_valid.write(
                "1 " + "valid_" + userid + " " + poss[i] + "%" + str(sess_id) + "\n"
            )
        for i in range(len(negs)):
            fp_valid.write(
                "0 " + "valid_" + userid + " " + negs[i] + "%" + str(sess_id) + "\n"
            )
    fp_valid.close()
    if os.path.isfile(valid_file_path):
        logger.info(f"Validation file {valid_file_path} successfully generated")
    else:
        raise FileNotFoundError(f"Error when generating {valid_file_path}")

def get_test_input(session, test_file_path):
    """Generate test file.

    Args:
        session (list): List of user session with user_id, clicks, all interactions.
        test_file_path (str): Path to file.
    """
    fp_test = open(test_file_path, "w", encoding="utf-8")
    clabel = 0
    for sess_id in range(len(session)):
        userid, _, imps = session[sess_id]
        for i in range(len(imps)):
            fp_test.write(
                str(clabel) + " " + "test_" + userid + " " + imps[i] + "%" + str(sess_id) + "\n"
            )
            clabel = clabel ^ 1
    fp_test.close()
    if os.path.isfile(test_file_path):
        logger.info(f"Test file {test_file_path} successfully generated")
    else:
        raise FileNotFoundError(f"Error when generating {test_file_path}")


def get_user_history(train_history, valid_history, user_history_path, test_history=None):
    """Generate user history file.

    Args:
        train_history (list): Train history.
        valid_history (list): Validation history
        user_history_path (str): Path to file.
    """
    fp_user_history = open(user_history_path, "w", encoding="utf-8")
    for userid in train_history:
        fp_user_history.write(
            "train_" + userid + " " + ",".join(train_history[userid]) + "\n"
        )
    for userid in valid_history:
        fp_user_history.write(
            "valid_" + userid + " " + ",".join(valid_history[userid]) + "\n"
        )
    if test_history is not None:
        for userid in test_history:
            fp_user_history.write(
                "test_" + userid + " " + ",".join(test_history[userid]) + "\n"
            )
    fp_user_history.close()
    if os.path.isfile(user_history_path):
        logger.info(f"User history file {user_history_path} successfully generated")
    else:
        raise FileNotFoundError(f"Error when generating {user_history_path}")


def _read_news(filepath, news_words, news_entities, tokenizer):
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        splitted = line.strip("\n").split("\t")
        news_words[splitted[0]] = tokenizer.tokenize(splitted[3].lower())
        news_entities[splitted[0]] = []
        for entity in json.loads(splitted[6]):
            news_entities[splitted[0]].append(
                (entity["SurfaceForms"], entity["WikidataId"])
            )
    return news_words, news_entities


def get_words_and_entities(train_news, valid_news, test_news=None):
    """Load words and entities

    Args:
        train_news (str): News train file.
        valid_news (str): News validation file.

    Returns: 
        dict, dict: Words and entities dictionaries.
    """
    news_words = {}
    news_entities = {}
    tokenizer = RegexpTokenizer(r"\w+")
    news_words, news_entities = _read_news(
        train_news, news_words, news_entities, tokenizer
    )
    news_words, news_entities = _read_news(
        valid_news, news_words, news_entities, tokenizer
    )
    if test_news is not None:
        news_words, news_entities = _read_news(
        test_news, news_words, news_entities, tokenizer
    )
    return news_words, news_entities


def _download_and_extract_globe(dest_path):
    url = "http://nlp.stanford.edu/data/glove.6B.zip"
    filepath = maybe_download(url=url, work_directory=dest_path)
    glove_path = os.path.join(dest_path, "glove")
    unzip_file(filepath, glove_path, clean_zip_file=False)
    return glove_path


def generate_embeddings(
    data_path,
    news_words,
    news_entities,
    train_entities,
    valid_entities,
    test_entities=None,
    max_sentence=10,
    word_embedding_dim=100,
):
    """Generate embeddings.

    Args:
        data_path (str): Data path.
        news_words (dict): News word dictionary.
        news_entities (dict): News entity dictionary.
        train_entities (str): Train entity file.
        valid_entities (str): Validation entity file.
        max_sentence (int): Max sentence size.
        word_embedding_dim (int): Word embedding dimension.

    Returns:
        str, str, str: File paths to news, word and entity embeddings.
    """
    embedding_dimensions = [50, 100, 200, 300]
    if word_embedding_dim not in embedding_dimensions:
        raise ValueError(
            f"Wrong embedding dimension, available options are {embedding_dimensions}"
        )

    logger.info("Downloading glove...")
    glove_path = _download_and_extract_globe(data_path)

    word_set = set()
    word_embedding_dict = {}
    entity_embedding_dict = {}

    logger.info(f"Loading glove with embedding dimension {word_embedding_dim}...")
    glove_file = "glove.6B." + str(word_embedding_dim) + "d.txt"
    fp_pretrain_vec = open(os.path.join(glove_path, glove_file), "r", encoding="utf-8")
    for line in fp_pretrain_vec:
        linesplit = line.split(" ")
        word_set.add(linesplit[0])
        word_embedding_dict[linesplit[0]] = np.asarray(list(map(float, linesplit[1:])))
    fp_pretrain_vec.close()

    logger.info("Reading train entities...")
    fp_entity_vec_train = open(train_entities, "r", encoding="utf-8")
    for line in fp_entity_vec_train:
        linesplit = line.split()
        entity_embedding_dict[linesplit[0]] = np.asarray(
            list(map(float, linesplit[1:]))
        )
    fp_entity_vec_train.close()

    logger.info("Reading valid entities...")
    fp_entity_vec_valid = open(valid_entities, "r", encoding="utf-8")
    for line in fp_entity_vec_valid:
        linesplit = line.split()
        entity_embedding_dict[linesplit[0]] = np.asarray(
            list(map(float, linesplit[1:]))
        )
    fp_entity_vec_valid.close()

    logger.info("Reading test entities...")
    fp_entity_vec_test = open(test_entities, "r", encoding="utf-8")
    for line in fp_entity_vec_test:
        linesplit = line.split()
        entity_embedding_dict[linesplit[0]] = np.asarray(
            list(map(float, linesplit[1:]))
        )
    fp_entity_vec_test.close()

    logger.info("Generating word and entity indexes...")
    word_dict = {}
    word_index = 1
    news_word_string_dict = {}
    news_entity_string_dict = {}
    entity2index = {}
    entity_index = 1
    for doc_id in news_words:
        news_word_string_dict[doc_id] = [0 for n in range(max_sentence)]
        news_entity_string_dict[doc_id] = [0 for n in range(max_sentence)]
        surfaceform_entityids = news_entities[doc_id]
        for item in surfaceform_entityids:
            if item[1] not in entity2index and item[1] in entity_embedding_dict:
                entity2index[item[1]] = entity_index
                entity_index = entity_index + 1
        for i in range(len(news_words[doc_id])):
            if news_words[doc_id][i] in word_embedding_dict:
                if news_words[doc_id][i] not in word_dict:
                    word_dict[news_words[doc_id][i]] = word_index
                    word_index = word_index + 1
                    news_word_string_dict[doc_id][i] = word_dict[news_words[doc_id][i]]
                else:
                    news_word_string_dict[doc_id][i] = word_dict[news_words[doc_id][i]]
                for item in surfaceform_entityids:
                    for surface in item[0]:
                        for surface_word in surface.split(" "):
                            if news_words[doc_id][i] == surface_word.lower():
                                if item[1] in entity_embedding_dict:
                                    news_entity_string_dict[doc_id][i] = entity2index[
                                        item[1]
                                    ]
            if i == max_sentence - 1:
                break

    logger.info("Generating word embeddings...")
    word_embeddings = np.zeros([word_index, word_embedding_dim])
    for word in word_dict:
        word_embeddings[word_dict[word]] = word_embedding_dict[word]

    logger.info("Generating entity embeddings...")
    entity_embeddings = np.zeros([entity_index, word_embedding_dim])
    for entity in entity2index:
        entity_embeddings[entity2index[entity]] = entity_embedding_dict[entity]

    news_feature_path = os.path.join(data_path, "doc_feature.txt")
    logger.info(f"Saving word and entity features in {news_feature_path}")
    fp_doc_string = open(news_feature_path, "w", encoding="utf-8")
    for doc_id in news_word_string_dict:
        fp_doc_string.write(
            doc_id
            + " "
            + ",".join(list(map(str, news_word_string_dict[doc_id])))
            + " "
            + ",".join(list(map(str, news_entity_string_dict[doc_id])))
            + "\n"
        )

    word_embeddings_path = os.path.join(
        data_path, "word_embeddings_5w_" + str(word_embedding_dim) + ".npy"
    )
    logger.info(f"Saving word embeddings in {word_embeddings_path}")
    np.save(word_embeddings_path, word_embeddings)

    entity_embeddings_path = os.path.join(
        data_path, "entity_embeddings_5w_" + str(word_embedding_dim) + ".npy"
    )
    logger.info(f"Saving word embeddings in {entity_embeddings_path}")
    np.save(entity_embeddings_path, entity_embeddings)

    return news_feature_path, word_embeddings_path, entity_embeddings_path
