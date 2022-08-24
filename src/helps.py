from genericpath import isdir
from importlib.resources import path
import torch.nn as nn
from torch import Tensor 
import torch
from typing import Union, Dict
from pathlib import Path
import shutil
import logging
import yaml
import random
import numpy as np

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

def log_cfg(cfg: Dict, prefix: str="cfg") -> None:
    """
    Write configuration to log.
    """
    logger = logging.getLogger(__name__)
    for key, value in cfg.items():
        if isinstance(value, Dict):
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








if __name__ == "__main__":
    # TEST 
    print(subsequent_mask(5))