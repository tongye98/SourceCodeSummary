from functools import partial
from torch import nn
from torch.optim import Optimizer
from typing import Generator
import torch
import logging
from src.helps import ConfigurationError
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, ExponentialLR

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
        # kwargs: lr; weight_decay; betas
        optimizer = torch.optim.Adam(parameters, **kwargs)
    elif optimizer_name == "sgd":
        # kwargs: lr; momentum; dampening; weight_decay; nestrov
        optimizer = torch.optim.SGD(parameters, **kwargs)
    else:
        raise ConfigurationError("Invalid optimizer.")
        
    logger.info("%s(%s)", optimizer.__class__.__name__, ", ".join([f"{key}={value}" for key,value in kwargs.items()]))
    return optimizer

def build_scheduler(train_cfg:dict, optimizer:Optimizer):
    """
    Create a learning rate scheduler if specified in train config and determine
    when a scheduler step should be executed.
    return:
        - scheduler: scheduler object
        - scheduler_step_at: "validation", "epoch", "step" or "none"
    """
    scheduler, scheduler_step_at = None, None
    scheduler_name = train_cfg.get("scheduling", None)
    assert scheduler_name in ["StepLR", "ExponentialLR", "ReduceLROnPlateau"], "Invalid scheduling."
    
    if scheduler_name == "ReduceLROnPlateau":
        mode = train_cfg.get("mode", "max")
        factor = train_cfg.get("factor", 0.5)
        patience = train_cfg.get("patience", 5)
        scheduler = ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience,
                        threshold=0.0001, threshold_mode='abs', eps=1e-8)
        scheduler_step_at = "validation"
    elif scheduler_name == "StepLR":
        step_size = train_cfg.get("step_size", 5)
        gamma = train_cfg.get("gamma", 0.1)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        scheduler_step_at = "epoch"  
    elif scheduler_name == "ExponentialLR":
        gamma  = train_cfg.get("gamma", 0.98)
        scheduler = ExponentialLR(optimizer, gamma=gamma)
        scheduler_step_at = "epoch"
    else:
        raise ConfigurationError("Invalid scheduling setting.")
    
    logger.info("Scheduler = %s", scheduler.__class__.__name__)
    return scheduler, scheduler_step_at
    