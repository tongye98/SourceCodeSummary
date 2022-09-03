# coding: utf-8
"""
This module for generating prediction of a model.
"""
import logging 
import torch 
from torch.utils.data import Dataset
from typing import Dict
from helps import load_config, load_model_checkpoint, parse_test_arguments, resolve_ckpt_path, write_list_to_file
import math
from datas import make_data_iter, load_data
from search import search
from model import build_model
import time
from metrics import Bleu, Meteor, Rouge
from pathlib import Path

logger = logging.getLogger(__name__)

def predict(model, data:Dataset, device:torch.device,
            n_gpu:int, compute_loss:bool=False, normalization:str="batch",
            num_workers:int=0, cfg:Dict=None, seed:int=980820):
    """
    Generate outputs(translations/summary) for the given data.
    If 'compute_loss' is True and references are given, also computes the loss.
    num_works: number of worers for 'collate_fn()' in data iterator.
    cfg: 'testing' section in yaml config file.
    return:
        - valid
    """
    (batch_size, batch_type, max_output_length, min_output_length,
     eval_metrics, beam_size, beam_alpha, n_best, return_attention, 
     return_prob, generate_unk, repetition_penalty, no_repeat_ngram_size) = parse_test_arguments(cfg)

    # FIXME what is return_prob
    if return_prob == "references":
        decoding_description = ""
    else:
        decoding_description = ("(Greedy decoding with " if beam_size < 2 else f"(Beam search with "
                                f"beam_size={beam_size}, beam_alpha={beam_alpha}, n_best={n_best}")
        decoding_description += (f"min_output_length={min_output_length}, "
                                 f"max_output_length={max_output_length}, return_prob={return_prob}, "
                                 f"genereate_unk={generate_unk}, repetition_penalty={repetition_penalty}, "
                                 f"no_repeat_ngram_size={no_repeat_ngram_size})")
    logger.info("Predicting %d examples...%s", len(data), decoding_description)

    assert batch_size >= n_gpu, "batch size must be bigger than n_gpu"

    data_iter = make_data_iter(dataset=data, sampler_seed=seed, shuffle=False, batch_type=batch_type,
                               batch_size=batch_size, num_workers=num_workers, device=device)

    model.eval()

    # Place holders for scores.
    valid_scores = {"loss":float("nan"), "ppl":float("nan"), "bleu":float(0),"meteor":float(0), "rouge-l":float(0)}
    total_nseqs = 0
    total_ntokens = 0
    total_loss = 0
    all_outputs = []
    valid_sentences_scores = []
    valid_attention_scores = [] 
    hyp_scores = None
    attention_scores = None

    for batch_data in data_iter:
        total_nseqs += batch_data.nseqs 

        # if compute_loss and batch_data.has_trg:
        if compute_loss: 
            assert model.loss_function is not None
            # don't track gradients during validation 
            with torch.no_grad():
                src_input = batch_data.src
                trg_input = batch_data.trg_input
                src_mask = batch_data.src_mask
                trg_mask = batch_data.trg_mask
                trg_truth = batch_data.trg
                batch_loss, log_probs = model(return_type="loss", src_input=src_input, trg_input=trg_input,
                   src_mask=src_mask, trg_mask=trg_mask, encoder_output = None, trg_truth=trg_truth)
                
                # sum over multiple gpus.
                # if normalization = 'sum', keep the same.
                batch_loss = batch_data.normalize(batch_loss, "sum", n_gpu)
                total_ntokens += batch_data.ntokens
                if return_prob == "references":
                    output = trg_truth
            
            total_loss += batch_loss.item()
            total_ntokens += batch_data.ntokens
        
        # if return_prob == "ref", then no search need. 
        # just look up the prob of the ground truth.
        if return_prob != "references":
            # run search as during inference to produce translations(summary)
            output, hyp_scores, attention_scores = search(model=model, batch_data=batch_data,
                                                          beam_size=beam_size, beam_alpha=beam_alpha,
                                                          max_output_length=max_output_length, 
                                                          min_output_length=min_output_length, n_best=n_best,
                                                          return_attention=return_attention, return_prob=return_prob,
                                                          generate_unk=generate_unk, repetition_penalty=repetition_penalty,
                                                          no_repeat_ngram_size=no_repeat_ngram_size)
            # output 
            #   greedy search: [batch_size, hyp_len/max_output_length] np.ndarray
            #   beam search: [batch_size*beam_size, hyp_len/max_output_length] np.ndarray
            # hyp_scores
            #   greedy search: [batch_size, hyp_len/max_output_length] np.ndarray
            #   beam search: [batch_size*n_best, hyp_len/max_output_length] np.ndarray
            # attention_scores 
            #   greedy search: [batch_size, steps/max_output_length, src_len] np.ndarray
            #   beam search: None
        
        all_outputs.extend(output)
        valid_sentences_scores.extend(hyp_scores if hyp_scores is not None else [])
        valid_attention_scores.extend(attention_scores if attention_scores is not None else [])

    assert total_nseqs == len(data)
    # FIXME all_outputs is a list of np.ndarray
    assert len(all_outputs) == len(data)*n_best

    if compute_loss:
        if normalization == "batch":
            normalizer = total_nseqs
        elif normalization == "tokens":
            normalizer = total_ntokens
        elif normalization == "none":
            normalizer = 1
        
        # avoid zero division
        assert normalizer > 0
        
        # normalized loss
        valid_scores["loss"] = total_loss / normalizer
        # exponent of token-level negative log likelihood
        valid_scores["ppl"] = math.exp(total_loss / total_ntokens)
    
    # decode ids back to str symbols (cut_off After eos, but eos itself is included. ) # FIXME: eos is not included.
    decoded_valid = model.trg_vocab.arrays_to_sentences(arrays=all_outputs, cut_at_eos=True)

    if return_prob == "references": # no evalution needed
        logger.info("Evaluation result (scoring) %s", 
                    ", ".join([f"{eval_metric}: {valid_scores[eval_metric]:6.2f}" for eval_metric in ["loss","ppl"]]))
        return (valid_scores, None, None, decoded_valid, None, None)

    # retrieve detokenized hypotheses and references
    valid_hypotheses = [data.tokenizer[data.trg_language].post_process(sentence, generate_unk=generate_unk) for sentence in decoded_valid]
    # valid_hypotheses -> list of strings
    # FIXME  add dataset trg function return -> list of strings
    valid_references = data.trg

    if True or data.has_trg: #TODO consider data has no trg situation.
        valid_hyp_1best = (valid_hypotheses if n_best == 1 else [valid_hypotheses[i] for i in range(0, len(valid_hypotheses), n_best)])
        assert len(valid_hyp_1best) == len(valid_references)

        predictions_dict = {k: [v.strip().lower()] for k,v in enumerate(valid_hyp_1best)}
        # 0: ['partitions a list of suite from a interval .']
        references_dict = {k: [v.strip().lower()] for k,v in enumerate(valid_references)}
        # 0: ['partitions a list of suite from a interval .']

        eval_metric_start_time = time.time()
        for eval_metric in eval_metrics:
            if eval_metric == "bleu":
                valid_scores[eval_metric], bleu_order = Bleu().corpus_bleu(hypotheses=predictions_dict, references=references_dict)
                # geometric mean of bleu scores
            elif eval_metric == "meteor":
                valid_scores[eval_metric] = Meteor().compute_score(gts=references_dict, res=predictions_dict)[0]
            elif eval_metric == "rouge-l":
                valid_scores[eval_metric] = Rouge().compute_score(gts=references_dict, res=predictions_dict)[0]
        eval_duration = time.time() - eval_metric_start_time
        eval_metrics_string = ", ".join([f"{eval_metric}:{valid_scores[eval_metric]:6.2f}" for eval_metric in 
                                          eval_metrics+["loss","ppl"] ])

        logger.info("Evaluation result(%s) %s, evaluation time: %.4f[sec]", "Beam Search" if beam_size > 1 else "Greedy Search",
                     eval_metrics_string, eval_duration)
    else:
        logger.info("No data trg truth provided.")
    
    return (valid_scores, bleu_order, valid_references, valid_hypotheses, decoded_valid,
            valid_sentences_scores, valid_attention_scores)

def test(cfg_file: str, ckpt_path:str, output_path:str=None, datasets:dict=None, save_scores:bool=True, save_attention:bool=False) -> None:
    """
    Main test function.
    Handles loading a model from checkpoint, generating translation, storing, and (optional) plotting attention.
    :param cfg_file: path to configuration file
    :param ckpt: path to model checkpoint to load
    :param output_path: path to output
    :param datasets: dict, to predict
    :param: sava_scores: whether to save scores
    :param: save_attention: whether to save attention visualization
    """ 
    cfg = load_config(Path(cfg_file))

    model_dir = cfg["training"].get("model_dir", None)
    load_model = cfg["training"].get("load_model", None)
    use_cuda = cfg["training"].get("use_cuda", False) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    n_gpu = torch.cuda.device_count() if use_cuda else 0
    num_workers = cfg["training"].get("num_workers", 0)
    normalization = cfg["training"].get("normalization","batch")
    seed = cfg["training"].get("random_seed", 980820)

    if datasets is None:
        train_data, dev_data, test_data, src_vocab, trg_vocab = load_data(data_cfg=cfg["data"])
        data_to_predict = {"dev": dev_data, "test":test_data}
    else:
        data_to_predict = {"dev":datasets["dev"], "test":datasets["test"]}
        src_vocab = datasets["src_vocab"]
        trg_vocab = datasets["trg_vocab"]
    
    # build an transformer(encoder-decoder) model
    model = build_model(model_cfg=cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)
    model.log_parameters_list()
    model.loss_function = (cfg["training"].get("loss","CrossEntropy"),cfg["training"].get("label_smoothing", 0.0))

    # when checkpoint is not specified, take latest(best) from model dir
    ckpt_path = resolve_ckpt_path(ckpt_path, load_model, model_dir)

    # load model checkpoint
    model_checkpoint = load_model_checkpoint(path=ckpt_path, device=device)

    #restore model and optimizer parameters
    model.load_state_dict(model_checkpoint["model_state"])
    if device.type == "cuda":
        model.to(device)
    
    # multi-gpu
    if n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # really test
    for dataset_name, dataset in data_to_predict.items():
        if dataset is not None:
            logger.info("Testing on %s set...", dataset_name)
            # FIXME compute_loss is set to true
            (valid_scores, bleu_order, valid_references, valid_hypotheses, decoded_valid,
            valid_sentences_scores, valid_attention_scores) = predict(model=model, data=dataset, device=device, n_gpu=n_gpu,
            compute_loss=True, normalization=normalization, num_workers=num_workers, cfg=cfg["testing"], seed=seed)
            if valid_hypotheses is not None:
                # save final model outputs.
                output_path = Path(f"{output_path}.{dataset_name}")
                write_list_to_file(file_path=output_path, array=valid_hypotheses)
                logger.info("Results saved to: %s.", output_path)
        else:
            logger.info(f"{dataset_name} is not exist!" )