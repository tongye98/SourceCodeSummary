from cProfile import label
import functools
from genericpath import isdir
from importlib.resources import path
from logging.handlers import TimedRotatingFileHandler
from os import cpu_count
from sys import maxsize
import torch.nn as nn
from torch import Tensor 
import torch
from typing import Union, Dict, Tuple, List
from pathlib import Path
import shutil
import logging
import yaml
import random
import numpy as np
import operator
from collections import Counter


def freeze_params(module: nn.Module) -> None:
    """
    freeze the parameters of this module,
    i.e. do not update them during training
    """
    for _, p in module.named_parameters():
        p.requires_grad = False

def subsequent_mask(size:int) -> Tensor:
    """
    Mask out subsequent positions (to prevent attending to future position)
    size: trg_len
    return:
        [1, size, size] (bool values)
    """
    ones = torch.ones(size,size, dtype=torch.bool)
    return torch.tril(ones, out=ones).unsqueeze(0) 

class ConfigurationError(Exception):
    """Custom exception for misspecifications of configuration."""

def load_config(path: Union[Path,str]="configs/transformer.yaml") -> Dict:
    if isinstance(path, str):
        path = Path(path)
    with path.open("r", encoding="utf-8") as yamlfile:
        cfg = yaml.safe_load(yamlfile)
    return cfg

def make_model_dir(model_dir: Path, overwrite: bool=False) -> Path:
    """
    Create a new directory for the model.
    return path to model directory.
    """
    model_dir = model_dir.absolute()
    if model_dir.is_dir():
        if not overwrite:
            raise FileExistsError(f"Model dir <model_dir> {model_dir} exists and overwrite is diabled.")
        shutil.rmtree(model_dir)
    model_dir.mkdir()
    return model_dir

def make_tensorboard_dir(tensorboard_dir: Path, overwrite: bool=False) -> None:
    """
    Create a tensorboard directory for the model.
    """
    tensorboard_dir = tensorboard_dir.absolute()
    if tensorboard_dir.is_dir():
        if not overwrite:
            raise FileExistsError(f"Tensorboard dir exists and overwrite is disable.")
        shutil.rmtree(tensorboard_dir)
    tensorboard_dir.mkdir()
    return None

def make_logger(log_dir: Path=None, mode:str="train") -> None:
    """
    Create a logger for logging the training/testing process.
    log_dir: path to file where log is stored as well
    mode: log file name. train test or translate.
    """
    logger = logging.getLogger("") # root logger
    # add handlers only once.
    if len(logger.handlers) == 0:
        logger.setLevel(level=logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
        if log_dir is not None:
            if log_dir.is_dir():
                log_file = log_dir / f"{mode}.log"
                fh = logging.FileHandler(log_file.as_posix())
                fh.setLevel(level=logging.DEBUG)
                fh.setFormatter(formatter)
                logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO) # FIXME change to INFO when have done.
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    logger.info("Hello! This is Tong Ye's Transformer!")
    return None

def log_cfg(cfg: dict, prefix: str="cfg") -> None:
    """
    Write configuration to log.
    """
    logger = logging.getLogger(__name__)
    for key, value in cfg.items():
        if isinstance(value, dict):
            p = ".".join([prefix, key])
            log_cfg(value, prefix=p)
        else:
            p = ".".join([prefix, key])
            logger.info("%42s : %s", p, value)

def set_seed(seed: int) -> None:
    """
    Set the random seed for modules torch, numpy and random.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)

def parse_train_arguments(train_cfg:dict) -> Tuple:
    logger = logging.getLogger(__name__)
    model_dir = Path(train_cfg["model_dir"])
    assert model_dir.is_dir(), f"{model_dir} not found!"
    tensorboard_dir = Path(train_cfg["tensorboard_dir"])

    use_cuda = train_cfg["use_cuda"] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    n_gpu = torch.cuda.device_count() if use_cuda else 0
    num_workers = train_cfg.get("num_workers", 0)
    if num_workers > 0:
        num_workers = min(cpu_count(), num_workers)
    
    normalization = train_cfg.get("normalization", "batch")
    if normalization not in ["batch","tokens"]:
        raise ConfigurationError("Invalid 'normalization' option.")
    
    loss_type = train_cfg.get("loss","CrossEntropy")
    label_smoothing = train_cfg.get("label_smoothing", 0.0)
    if loss_type not in ["CrossEntropy"]:
        raise ConfigurationError("Invalid 'loss'.")
    
    learning_rate_min = train_cfg.get("learning_rate_min", 1.0e-8)
    
    keep_best_ckpts = int(train_cfg.get("keep_best_ckpts",5))

    logging_freq = train_cfg.get("logging_freq", 100)
    validation_freq = train_cfg.get("validation_freq", 1000)
    log_valid_sentences = train_cfg.get("log_valid_sentences", [0,1,2])

    early_stopping_metric = train_cfg.get("early_stopping_metric", "bleu").lower()
    if early_stopping_metric not in ["acc","loss","ppl","bleu"]:
        raise ConfigurationError("Invalid setting for 'early_stopping_metric'.")
    
    shuffle = train_cfg.get("shuffle", True)
    epochs = train_cfg.get("epochs", 100)
    max_updates = train_cfg.get("max_updates", np.inf)
    batch_size = train_cfg.get("batch_size", 32)
    batch_type = train_cfg.get("batch_type", "sentence")
    if batch_type not in ["sentence", "token"]:
        raise ConfigurationError("Invalid 'batch_type'. ")
    random_seed = train_cfg.get("random_seed", 980820)  

    load_model = train_cfg.get("load_model", None)
    if load_model is not None:
        load_model = Path(load_model)
        assert load_model.is_file()

    reset_best_ckpt = train_cfg.get("reset_best_ckpt", False)
    reset_scheduler = train_cfg.get("reset_scheduler", False)
    reset_optimizer = train_cfg.get("reset_optimizer", False)
    reset_iter_state = train_cfg.get("rest_iter_state", False)

    return(model_dir, tensorboard_dir ,loss_type, label_smoothing,
           normalization, learning_rate_min, keep_best_ckpts,
           logging_freq, validation_freq, log_valid_sentences,
           early_stopping_metric, shuffle, epochs, max_updates,
           batch_size, batch_type, random_seed,
           device, n_gpu, num_workers,load_model,
           reset_best_ckpt, reset_scheduler,
           reset_optimizer, reset_iter_state)

def parse_test_arguments(test_cfg:dict) -> Tuple:
    """Parse test args."""
    logger = logging.getLogger(__name__)
    batch_size = test_cfg.get("batch_size", 64)
    batch_type = test_cfg.get("batch_type", "sentence")
    if batch_type not in ["sentence", "token"]:
        raise ConfigurationError("Invalid batch type option.")
    if batch_type == "setence" and batch_size > 1000:
        logger.warning("test batch size is too huge.")
    
    max_output_length = test_cfg.get("max_output_length", 30)
    min_output_length = test_cfg.get("min_output_length", 1)

    if "eval_metrics" in test_cfg.keys():
        eval_metrics = [metric.strip().lower() for metric in test_cfg["eval_metrics"]]
    
    for eval_metric in eval_metrics:
        if eval_metric not in ["bleu", "meteor", "rouge-l"]:
            raise ConfigurationError("eval metric is Invalid.")
    
    # beam search options
    n_best = test_cfg.get("n_best", 1)
    beam_size = test_cfg.get("beam_size", 1)
    beam_alpah = test_cfg.get("beam_alpha", -1)
    assert beam_size > 0 and n_best > 0 and n_best <= beam_size, "Invalid beam search options."

    # control options
    return_attention = test_cfg.get("return_attention", False)
    # FIXME what is return prob?
    return_prob = test_cfg.get("return_prob", "none")
    if return_prob not in ["hypotheses","references","none"]:
        raise ConfigurationError("Invalid return_prob")
    generate_unk = test_cfg.get("generate_unk", True)
    repetition_penalty = test_cfg.get("repetition_penalty", -1)
    if repetition_penalty < 1 and repetition_penalty != -1:
        raise ConfigurationError("Invalid repetition_penalty.")
    no_repeat_ngram_size = test_cfg.get("no_repeat_ngram_size", -1)

    return (batch_size, batch_type, max_output_length, min_output_length,
            eval_metrics, beam_size, beam_alpah, n_best, return_attention, 
            return_prob, generate_unk, repetition_penalty, no_repeat_ngram_size)

def load_model_checkpoint(path:Path, device:torch.device) -> Dict:
    """
    Load model from saved model checkpoint
    """
    logger = logging.getLogger(__name__)
    assert path.is_file(), f"model checkpoint {path} not found!"
    model_checkpoint = torch.load(path.as_posix(), map_location=device)
    logger.info("Load model from %s.", path.resolve())
    return model_checkpoint

def symlink_update(target:Path, link_name: Path):
    """
    find the file that the symlink currently points to, sets it to 
    the new target, and return the previous target if it exists.
    target: a path to a file that we want the symlink to point to.
        no parent dir, filename only, i.e. "1000.ckpt"
    link_name: the name of the symlink that we want to update.
        link_name with parent dir, i.e. "models/my_model/best.ckpt"
    """
    if link_name.is_symlink():
        current_last = link_name.resolve()
        link_name.unlink()
        link_name.symlink_to(target)
        return current_last
    link_name.symlink_to(target)
    return None 

def delete_ckpt(path:Path) -> None:
    logger = logging.getLogger(__name__)
    try:
        logger.info("Delete %s", path.as_posix())
        path.unlink()
    except FileNotFoundError as error:
        logger.warning("Want to delete old checkpoint %s"
            "but file does not exist. (%s)", path, error)

def write_validation_output_to_file(path:Path, array: List[str]) -> None:
    """
    Write list of strings to file.
    array: list of strings.
    """
    with path.open("w", encoding="utf-8") as fg:
        for entry in array:
            fg.write(f"{entry}\n")

def read_list_from_file(path:Path):
    """
    Read list of string from file.
    return list of strings.
        i.e. ['hello world', 'i am xxx']
    """
    return [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]

def flatten(array):
    # flatten a nested 2D list.
    return functools.reduce(operator.iconcat, array, [])

def sort_and_cut(counter, max_size:int, min_freq:int) -> List[str]:
    """
    Cut counter to most frequent, sorted numerically and alphabetically.
    return: list of valid tokens
    """
    if min_freq > -1:
        counter = Counter({t: c for t, c in counter.items() if c >= min_freq})
    
    # sort by frequency, then alphabetically
    tokens_and_frequencies = sorted(counter.items(),key=lambda tup: tup[0])
    tokens_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    # cut off
    vocab_tokens = [i[0] for i in tokens_and_frequencies[:max_size]]
    assert len(vocab_tokens) <= max_size, "vocab tokens must <= max_size."
    return vocab_tokens

def log_data_info(train_data, dev_data, test_data,
                  src_vocab, trg_vocab) -> None:
    """
    Log statistics of data and vocabulary.
    """
    logger = logging.getLogger(__name__)
    logger.info("Train dataset: %s", train_data)
    logger.info("Valid dataset: %s", dev_data)
    logger.info(" Test dataset: %s", test_data)

    if train_data:
        src = train_data.get_item(idx=0, language=train_data.src_language)
        trg = train_data.get_item(idx=0, language=train_data.trg_language)
        src_print = "\n\t[SRC] " + " ".join(src)
        trg_print = "\n\t[TRG] " + " ".join(trg)
        logger.info("First training example: %s%s", src_print, trg_print)


    logger.info("Number of unique Src tokens (vocab size): %d", len(src_vocab))
    logger.info("Number of unique Trg tokens (vocab size): %d", len(trg_vocab))

    logger.info("First 10 Src tokens: %s", src_vocab.log_vocab(10))
    logger.info("First 10 Trg tokens: %s", trg_vocab.log_vocab(10))

def write_list_to_file(file_path:Path, array:List[str]) -> None:
    """
    write list of str to file.
    """
    with file_path.open("w", encoding="utf-8") as fg:
        for item in array:
            fg.write(f"{item}\n")

if __name__ == "__main__":
    # TEST 
    print(subsequent_mask(5))
    sentences = read_list_from_file(Path('data/test_datasets/train.txt'))
    print(sentences)
    sentences = [sentence.split(" ") for sentence in sentences]
    print(sentences)
    print(flatten(sentences))