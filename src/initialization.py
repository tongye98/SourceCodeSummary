# coding: utf-8
"""
Implements custom initialization
"""
import torch 
from torch import Tensor, nn 
from helps import ConfigurationError 

def Initialize_model(model: nn.Module, model_cfg:dict,
                    src_pad_index: int, trg_pad_index: int) -> None:
    """
    All initializer configuration is part of the 'model' section of the configuration file.
    The main initializer is set using the 'initializer' key.
        possible values are 'xavier', 'uniform', 'normal' or 'zeros'('xavier' is the default)
    when an initializer is set to 'uniform',
        then 'init_weight' sets the range for the values (-init_weight, init_weight)
    when an initializer is set to 'normal',
        then 'init_weight' sets the standard deviation for the weights (with mean 0).

    """
    init = model_cfg.get("initializer", "xavier_uniform")
    init_gain = float(model_cfg.get("init_gain", 1.0)) # for xavier
    init_weight = float(model_cfg.get("init_weight", 0.01)) # for uniform and normal

    embed_init = model_cfg.get("embed_initializer", "normal")
    embed_init_gain = float(model_cfg.get("embed_init_gain",1.0))
    embed_init_weight = float(model_cfg.get("embed_init_weight", 0.01))

    bias_init = model_cfg.get("bias_initializer", "zeros")
    bias_init_weight = float(model_cfg.get("bias_init_weight", 0.01))

    def parse_init(init:str, weight: float, gain:float):
        weight = float(weight)
        assert weight > 0.0, "Incorrect init weight"
        if init.lower() == "xavier_uniform":
            return lambda p: nn.init.xavier_uniform_(p, gain=gain)
        elif init.lower() == "xavier_normal":
            return lambda p: nn.init.xavier_normal_(p, gain=gain)
        elif init.lower() == "uniform":
            return lambda p: nn.init.uniform_(p, a=-weight, b=weight)
        elif init.lower() == "normal":
            return lambda p: nn.init.normal_(p, mean=0.0, std=weight)
        elif init.lower() == "zeros":
            return lambda p: nn.init.zeros_(p)
    
    init_fn = parse_init(init=init, weight=init_weight, gain=init_gain)
    embed_init_fn = parse_init(init=embed_init, weight=embed_init_weight, gain=embed_init_gain)
    bias_init_fn = parse_init(init=bias_init, weight=bias_init_weight, gain=init_gain)

    with torch.no_grad():
        for name, param in model.named_parameters():
            if "embed" in name:
                embed_init_fn(param)
            elif "bias" in name:
                bias_init_fn(param)
            elif len(param.size()) > 1:
                init_fn(param)

        # zero out paddings
        model.src_embed.lut.weight.data[src_pad_index].zero_()
        model.trg_embed.lut.weight.data[trg_pad_index].zero_()