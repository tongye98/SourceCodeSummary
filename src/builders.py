from functools import partial
from torch import nn
from torch.optim import Optimizer
from typing import Generator
import torch
import logging
from src.helps import ConfigurationError
logger = logging.getLogger(__name__)
def build_gradient_clipper(train_cfg: dict):
    """
    Define the function for gradient clipping as specified in configuration.
    Current Options:
        clip_grad_val: clip the gradients if they exceed value
            torch.nn.utils.clip_grad_value_
        clip_grad_norm: clip the gradients if they norm exceeds this value
            torch.nn.utils.clip_grad_norm_
    """

    clip_grad_function = None
    if "clip_grad_val" in train_cfg.keys():
        clip_grad_function = partial(nn.utils.clip_grad_value_, clip_value=train_cfg["clip_grad_val"])
    elif "clip_grad_norm" in train_cfg.keys():
        clip_grad_function = partial(nn.utils.clip_grad_norm_, max_norm=train_cfg["clip_grad_norm"])
    return clip_grad_function

def build_optimizer(train_cfg:dict, parameters: Generator)  -> Optimizer:
    """
    Create an optimzer for the given parameters as specified in config.
    """
    optimizer_name = train_cfg.get("optimizer",'adam').lower()
    kwargs = {"lr": train_cfg.get("learning_rate", 3e-4), "weight_decay":train_cfg.get("weight_decay",0)}
    if optimizer_name == "adam":
        kwargs["betas"] = train_cfg.get("adam_betas", (0.9, 0.999))
        optimizer = torch.optim.Adam(parameters, **kwargs)
    else:
        raise ConfigurationError("Invalid optimizer.")
        
    logger.info("%s(%s)", optimizer.__class__.__name__, ", ".join([f"{key}={value}" for key,value in kwargs.items()]))
    return optimizer
