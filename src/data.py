# coding: utf-8
"""
Data module
"""
import logging
import torch 
from tokenizers import build_tokenizer
from datasets import build_dataset
from vocabulary import build_vocab
from helps import ConfigurationError, log_data_info
from torch.utils.data import Dataset, Sampler, DataLoader
from torch.utils.data import SequentialSampler, RandomSampler, BatchSampler
from typing import List, Union, Tuple, Iterator, Iterable

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

    dataset_type = data_cfg.get("dataset_type", "plain")

    # train data 
    train_data = None 
    if train_data_path is not None:
        logger.info("Loading train dataset...")
        train_data = build_dataset(dataset_type=dataset_type,path=train_data_path,
                                   src_language=src_language, trg_language=trg_language)
    
    # build vocabulary
    logger.info("Building vocabulary...")
    src_vocab, trg_vocab = build_vocab(data_cfg, dataset=train_data)

    # dev data
    dev_data = None
    if dev_data_path is not None:
        logger.info("Loading dev dataset...")
        dev_data = build_dataset(dataset_type=dataset_type, path=dev_data_path,
                                 src_language=src_language, trg_language=trg_language)
    
    # test data
    if test_data_path is not None:
        logger.info("Load test dataset...")
        test_data = build_dataset(dataset_type=dataset_type, path=test_data_path,
                                 src_language=src_language, trg_language=trg_language)
    
    logger.info("Dataset has loaded.")
    log_data_info(train_data, dev_data, test_data, src_vocab, trg_vocab)

    return train_data, dev_data, test_data, src_vocab, trg_vocab

def make_data_iter(dataset:Dataset, sampler_seed, shuffle, batch_type,
                   batch_size,num_workers) -> DataLoader:
    """
    Return a torch DataLoader for a torch Dataset.
    """
    assert isinstance(dataset, Dataset), "For pytorch, dataset is based on torch.utils.data.Dataset"

    if dataset.split_mode == "train" and shuffle is True:
        generator = torch.Generator()
        generator.manual_seed(sampler_seed)
        sampler = RandomSampler(dataset, replacement=False,generator=generator)
    else:
        sampler = SequentialSampler(dataset)
    
    if batch_type == "sentence":
        batch_sampler = SentenceBatchSampler(sampler, batch_size=batch_size, drop_last=False)
    elif batch_type == "token":
        batch_sampler = TokenBatchSampler(sampler, batch_size=batch_size, drop_last=False)
    else:
        raise ConfigurationError("Invalid batch_type")
    
    return DataLoader(dataset, batch_sampler=batch_sampler, shuffle=False, num_workers=num_workers,
                      pin_memory=True, collate_fn=collate_fn)

class SentenceBatchSampler(BatchSampler):
    def __init__(self, sampler: Union[Sampler[int], Iterable[int]], batch_size: int, drop_last: bool) -> None:
        super().__init__(sampler, batch_size, drop_last)
    
    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch
    
    def __len__(self) -> int:
        return super().__len__()

class TokenBatchSampler(BatchSampler):
    def __init__(self, sampler: Union[Sampler[int], Iterable[int]], batch_size: int, drop_last: bool) -> None:
        super().__init__(sampler, batch_size, drop_last)
    
    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch
    
    def __len__(self) -> int:
        return super().__len__()

def collate_fn(batch: List[Tuple]):
    """
    Custom collate function.
    DataLoader每次迭代的返回值就是collate_fn的返回值.
    """
    batch = [ (src, trg) for (src, trg) in batch]
    # you can stack batch and any operation on batch 
    # FIXME make batch to tensor.
    return batch