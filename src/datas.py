# coding: utf-8
"""
Data module
"""
import logging
import torch 
from src.tokenizers import build_tokenizer
from src.datasets import build_dataset
from src.vocabulary import Vocabulary, build_vocab
from src.helps import ConfigurationError, log_data_info, make_src_map, align
from torch.utils.data import Dataset, Sampler, DataLoader
from torch.utils.data import SequentialSampler, RandomSampler, BatchSampler
from typing import List, Union, Tuple, Iterator, Iterable
from functools import partial
from src.batch import Batch

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
        train_data = build_dataset(dataset_type=dataset_type,path=train_data_path, split_mode="train",
                                   src_language=src_language, trg_language=trg_language, tokenizer=tokenizer)
    
    # build vocabulary
    logger.info("Building vocabulary...")
    src_vocab, trg_vocab = build_vocab(data_cfg, dataset=train_data)

    train_data.tokernized_data_to_ids(src_vocab, trg_vocab)

    # dev data
    dev_data = None
    if dev_data_path is not None:
        logger.info("Loading dev dataset...")
        dev_data = build_dataset(dataset_type=dataset_type, path=dev_data_path, split_mode="dev",
                                 src_language=src_language, trg_language=trg_language, tokenizer=tokenizer)
        dev_data.tokernized_data_to_ids(src_vocab, trg_vocab)

    # test data
    if test_data_path is not None:
        logger.info("Loading test dataset...")
        test_data = build_dataset(dataset_type=dataset_type, path=test_data_path, split_mode="test",
                                 src_language=src_language, trg_language=trg_language, tokenizer=tokenizer)
        test_data.tokernized_data_to_ids(src_vocab, trg_vocab)

    logger.info("Dataset has loaded.")
    log_data_info(train_data, dev_data, test_data, src_vocab, trg_vocab)

    return train_data, dev_data, test_data, src_vocab, trg_vocab

def make_data_iter(dataset:Dataset, sampler_seed, shuffle, batch_type,
                   batch_size, num_workers) -> DataLoader:
    """
    Return a torch DataLoader for a torch Dataset.
    """
    assert isinstance(dataset, Dataset), "For pytorch, dataset is based on torch.utils.data.Dataset"

    if dataset.split_mode == "train" and shuffle is True:
        generator = torch.Generator()
        generator.manual_seed(sampler_seed)
        sampler = RandomSampler(dataset, replacement=False, generator=generator)
    else:
        sampler = SequentialSampler(dataset)
    
    if batch_type == "sentence":
        batch_sampler = SentenceBatchSampler(sampler, batch_size=batch_size, drop_last=False)
    elif batch_type == "token":
        batch_sampler = TokenBatchSampler(sampler, batch_size=batch_size, drop_last=False)
    else:
        raise ConfigurationError("Invalid batch_type")
    
    return DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers,
                      pin_memory=True, collate_fn=collate_fn)

def collate_fn(batch: List[Tuple]) -> Batch:
    """
    Custom collate function.
    Note: you can stack batch and any operation on batch.
    DataLoader every iter result is collate_fn's return value -> Batch.
    :param batch [(src,trg),(src,trg),...]
    Note: for copy mechanism, need src_vocabs, source_maps, alignments
    """
    src_list, trg_list = zip(*batch) # src_list: Tuple[List[id]]
    assert len(src_list) == len(trg_list)

    # # for copy mechanism
    # batch_size = len(batch)

    # src_vocabs = []
    # source_maps = []
    # alignments = []
    # for id in range(batch_size):
    #     vocab = Vocabulary(tokens=src_list[id], has_bos_eos=False)
    #     src_vocabs.append(vocab)
    #     src_map = torch.tensor([vocab.lookup(token) for token in src_list[id]])
    #     source_maps.append(src_map)

    #     alignment = torch.tensor([vocab.lookup(token) for token in trg_list[id]] + [0]) # [0] for eos == trg_len
    #     alignments.append(alignment)

    # source_maps = make_src_map(source_maps)
    # # source_maps [batch_size, src_len, extra_words]
    # assert source_maps.size(1) == src.size(1)

    # alignments = align(alignments)
    # # alignments [batch_size, original_target_length+1]  no bos, but has eos
    
    return Batch(src_list, trg_list)


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