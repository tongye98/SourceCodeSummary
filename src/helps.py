import os
import functools
import torch
import shutil
import logging
import yaml
import random
import numpy as np
import operator
import torch.nn as nn
from pathlib import Path
from os import cpu_count
from torch import Tensor 
from collections import Counter
from typing import Union, Dict, Tuple, List
from src.constants import EOS_TOKEN, PAD_TOKEN, UNK_ID

logger = logging.getLogger(__name__)

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
                if log_file.exists():
                    assert False, "log file exists."
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
    model_dir = Path(train_cfg["model_dir"])
    assert model_dir.is_dir(), f"{model_dir} not found!"

    use_cuda = train_cfg["use_cuda"] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    n_gpu = torch.cuda.device_count() if use_cuda else 0
    logger.info("*"*20 + "{} GPU".format(n_gpu) + "*"*20)
    num_workers = train_cfg.get("num_workers", 0)
    if num_workers > 0:
        num_workers = min(cpu_count(), num_workers)
    
    normalization = train_cfg.get("normalization", "batch")
    if normalization not in ["batch","tokens"]:
        raise ConfigurationError("Invalid 'normalization' option.")
    
    loss_function = train_cfg.get("loss", "CrossEntropy")
    label_smoothing = train_cfg.get("label_smoothing", 0.0)
    if loss_function not in ["CrossEntropy"]:
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

    return(model_dir, loss_function, label_smoothing,
           normalization, learning_rate_min, keep_best_ckpts,
           logging_freq, validation_freq, log_valid_sentences,
           early_stopping_metric, shuffle, epochs, max_updates,
           batch_size, batch_type, random_seed,
           device, n_gpu, num_workers,load_model,
           reset_best_ckpt, reset_scheduler,
           reset_optimizer, reset_iter_state)

def parse_test_arguments(test_cfg:dict) -> Tuple:
    """Parse test args."""
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
    return_prob = test_cfg.get("return_prob", "none")
    if return_prob not in ["hypotheses","references","none"]:
        raise ConfigurationError("Invalid return_prob")
    generate_unk = test_cfg.get("generate_unk", True)
    repetition_penalty = test_cfg.get("repetition_penalty", -1)

    return (batch_size, batch_type, max_output_length, min_output_length,
            eval_metrics, beam_size, beam_alpah, n_best, return_attention, 
            return_prob, generate_unk, repetition_penalty)

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
        src = train_data.tokernized_data[train_data.src_language][0]
        trg = train_data.tokernized_data[train_data.trg_language][0]
        src_print = "\n\t[SRC] " + " ".join(src)
        trg_print = "\n\t[TRG] " + " ".join(trg)
        logger.info("First training example: %s%s", src_print, trg_print)

        src_id = train_data.tokernized_data_ids[train_data.src_language][0]
        trg_id = train_data.tokernized_data_ids[train_data.trg_language][0]
        src_id_print = "\n\t[SRC ID] " + str(src_id)
        trg_id_print = "\n\t[TRG ID] " + str(trg_id)
        logger.info("First training example id: %s%s", src_id_print, trg_id_print)

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

def tile(x: Tensor, count: int, dim : int=0) -> Tensor:
    """
    Tiles x on dimension 'dim' count times. Used for beam search.
    i.e. [a,b] --count=3--> [a,a,a,b,b,b]
    :param: x [batch_size, src_len, model_dim]
    return tiled tensor
    """
    assert dim == 0
    out_size = list(x.size()) # [batch_size, src_len, model_dim]
    out_size[0] = out_size[0] * count # [batch_size*count, src_len, model_dim]
    batch_size = x.size(0)
    x = x.view(batch_size, -1).transpose(0,1).repeat(count, 1).transpose(0,1).contiguous().view(*out_size)
    return x

def resolve_ckpt_path(ckpt_path:str, load_model:str, model_dir:Path) -> Path:
    """
    Resolve checkpoint path
    First choose ckpt_path, then choos load_model, 
    then choose model_dir/best.ckpt, final choose model_dir/latest.ckpt
    """
    logger = logging.getLogger(__name__)
    if ckpt_path is None:
        if load_model is None:
            if (model_dir / "best.ckpt").is_file():
                ckpt_path = model_dir / "best.ckpt"
            else:
                logger.warning("No ckpt_path, no load_model, no best_model, Please Check!")
                ckpt_path = model_dir / "latest.ckpt"
        else:
            ckpt_path = Path(load_model)
    return Path(ckpt_path)

def generate_relative_position_matrix(length, max_relative_position, use_negative_distance):
    """
    Generate the clipped relative position matrix.
    """
    range_vec = torch.arange(length)
    range_matrix = range_vec.unsqueeze(1).expand(-1, length).transpose(0,1)
    distance_matrix = range_matrix - range_matrix.transpose(0, 1)
    distance_mat_clipped = torch.clamp(distance_matrix, min=-max_relative_position, max=max_relative_position)

    if use_negative_distance:
        final_matrix = distance_mat_clipped + max_relative_position
    else:
        final_matrix = torch.abs(distance_mat_clipped)

    return final_matrix

def check_retrieval_cfg(retrieval_cfg: dict) -> None:
    """
    This function is used to validate that the merged retrieval config is valid.
    """
    assert retrieval_cfg["type"] in ["no_retriever", "static_retriever", "dynamic_retriever"], \
        "retrieval type {} is not supported currently.".format(retrieval_cfg["type"])
    
    if retrieval_cfg["type"] in ["static_retriever", "dynamic_retriever"]:
        for key in ["top_k", "kernel"]:
            assert retrieval_cfg[key] is not None, "{} is needed for {}".format(key, retrieval_cfg["type"])
        
        for key in ["index_path", "token_map_path"]:
            assert retrieval_cfg[key] is not None, "{} is needed for {}".format(key, retrieval_cfg["type"])
            path = retrieval_cfg[key]
            assert os.path.exists(path), "{} does not exist.".format(path)
        
        if retrieval_cfg["type"] == "static_retriever":
            for key in ["mixing_weight", "bandwidth"]:
                assert retrieval_cfg[key] is not None, "{} is needed in {}".format(key, retrieval_cfg["type"])
        
        if retrieval_cfg["type"] == "dynamic_retriever":
            assert retrieval_cfg["embedding_path"] is not None, "{} is needed in {}".format("embedding_path", retrieval_cfg["type"])
            path = retrieval_cfg["embedding_path"] 
            assert os.path.exists(path), "{} does not exist.".format(path)

def retrieval_accuracy(token_indices, trg_truth):
    """
    token_indices [batch_size*trg_len, top_k] id
    trg_truth: [batch_size, trg_len]
    """
    hits = 0.
    first_hits = 0.
    all = 0
    trg_truth = trg_truth.view(-1, 1) 
    # trg_truth [batch_size*trg_len, 1]
    for id, trg_id in enumerate(trg_truth):
        if trg_id.item() == 1 or trg_id.item() == 3: # pad id and eos id
            continue 
        all += 1
        if trg_id in token_indices[id]:
            hits += 1
        if trg_id == token_indices[id][0]:
            first_hits += 1
    # logger.info("hit accuracy = {}".format(hits/all))
    return hits, first_hits, all

def actually_help_analysis(analysis, trg_truth):
    """
    analysis: dict
    trg_truth: [batch_size, trg_len]
    """
    token_indices = analysis["token_indices"]
    model_based_distribution = analysis["model_based_distribution"]
    example_based_distribution = analysis["example_based_distribution"]
    mixed_distribution = analysis["mixed_distribution"]
    # distribution [batch_size*trg_len, vocab_size]

    batch_size, trg_len = trg_truth.size()
    vocab_size = model_based_distribution.size(-1)
    trg_truth = trg_truth.view(-1)
    # trg_truth [batch_size*trg_len]

    model_probs, model_words = torch.max(model_based_distribution, dim=-1)
    # probs [batch_size*trg_len]
    # words [batch_size*trg_len]

    example_probs, example_words = torch.max(example_based_distribution,dim=-1)

    mixed_probs, mixed_words = torch.max(mixed_distribution, dim=-1)
    # step1 original is right, after retriever still right
    # step2 original is right, after retriever become false
    # step3 original is false, after retriever become right
    # step4 original is false, after retriever become false
    
    token_num = 0
    model_true_mix_true_num = 0
    model_true_mix_false_num = 0
    model_false_mix_true_num = 0
    model_false_mix_false_num = 0

    for id in range(batch_size*trg_len):
        if trg_truth[id] == 1 or trg_truth[id] == 3: # pad id eos id
            continue 

        token_num += 1

        if model_words[id] == trg_truth[id] and mixed_words[id] == trg_truth[id]:
            model_true_mix_true_num += 1

        if model_words[id] == trg_truth[id] and mixed_words[id] != trg_truth[id]:
            model_true_mix_false_num += 1
        
        if model_words[id] != trg_truth[id] and mixed_words[id] == trg_truth[id]:
            model_false_mix_true_num += 1
        
        if model_words[id] != trg_truth[id] and mixed_words[id] != trg_truth[id]:
            model_false_mix_false_num += 1
    
    help_analysis = dict()
    help_analysis["token_num"] = token_num
    help_analysis["model_true_mix_true_num"] = model_true_mix_true_num
    help_analysis["model_true_mix_false_num"] = model_true_mix_false_num
    help_analysis["model_false_mix_true_num"] = model_false_mix_true_num
    help_analysis["model_false_mix_false_num"] = model_false_mix_false_num
    return help_analysis
