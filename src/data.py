# coding: utf-8
"""
Data module
"""
from dataclasses import dataclass
import logging
import torch 

logger = logging.getLogger(__name__)

def load_data(data_cfg: dict):
    """
    Load train, dev and test data as specified in configuration.
    Vocabularies are created from the training dataset with a limit of 'vocab_limit' tokens
    and a minimum token frequency of "vocab_min_freq".
    The training data is cut off to include sentences up to "max_sentence_length"
    on source and target side.
    return:
        - src_vocab: source vocabulary
        - trg_vocab: traget vocabulary
        - train_data: training dataset
        - dev_data: development dataset
        - test_data: test dataset
    """
    src_cfg = data_cfg["src"]
    trg_cfg = data_cfg["trg"]

    # Load data from files
    src_language = src_cfg["language"]
    trg_language = trg_cfg["language"]
    train_data_path = data_cfg.get("train_data_path", None)
    dev_data_path = data_cfg.get("dev_data_path", None)
    test_data_path = data_cfg.get("test_data_path", None)

    assert train_data_path is not None
    assert dev_data_path is not None 
    # assert test_data_path is not None 
    
    # build tokenizer
    logger.info("Build tokenizer...")
    tokenizer = build_tokenizer(data_cfg)

    # train data 
    train_data = None 
    if train_data_path is not None:
        logger.info("Loading train dataset...")
        train_data = build_dataset()
    
    # build vocabulary
    logger.info("Building vocabulary...")
    src_vocab, trg_vocab = build_vocab(data_cfg, dataset=train_data)

    # dev data
    dev_data = None
    if dev_data_path is not None:
        logger.info("Loading dev dataset...")
        dev_data = build_dataset()
    
    # test data
    if test_data_path is not None:
        logger.info("Load test dataset...")
        test_data = build_dataset()
    
    logger.info("Dataset has loaded.")
    log_data_info()
    return train_data, dev_data, test_data, src_vocab, trg_vocab

def make_data_iter():
    pass 

def collate_fn():
    pass