from cProfile import label
from genericpath import isdir
from importlib.resources import path
from logging.handlers import TimedRotatingFileHandler
from os import cpu_count
import torch.nn as nn
from torch import Tensor 
import torch
from typing import Union, Dict, Tuple
from pathlib import Path
import shutil
import logging
import yaml
import random
import numpy as np

from src.training import train

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

def make_model_dir(model_dir:Path, overwrite:bool=False) -> Path:
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
    sh.setLevel(logging.INFO)
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
    
    load_model = train_cfg.get("load_model",None)
    if load_model is not None:
        load_model = Path(load_model)
        assert load_model.is_file()

    reset_best_ckpt = train_cfg.get("reset_best_ckpt", False)
    reset_scheduler = train_cfg.get("reset_scheduler", False)
    reset_optimizer = train_cfg.get("reset_optimizer", False)
    reset_iter_state = train_cfg.get("rest_iter_state", False)

    return(model_dir, loss_type, label_smoothing,
           normalization, learning_rate_min, keep_best_ckpts,
           logging_freq, validation_freq, log_valid_sentences,
           early_stopping_metric, shuffle, epochs, max_updates,
           batch_size, batch_type, random_seed,
           device, n_gpu, num_workers,load_model,
           reset_best_ckpt, reset_scheduler,
           reset_optimizer, reset_iter_state)

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

if __name__ == "__main__":
    # TEST 
    print(subsequent_mask(5))