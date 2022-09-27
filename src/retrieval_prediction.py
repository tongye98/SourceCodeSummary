# coding: utf-8
"""
This module for generating prediction of a model.
"""
import logging 
import torch 
from torch.utils.data import Dataset
from typing import Dict
from src.helps import collapse_copy_scores, load_config, load_model_checkpoint, parse_test_arguments, make_logger
from src.helps import resolve_ckpt_path, write_list_to_file, cut_off
import math
from src.datas import make_data_iter, load_data
from src.search import search
from src.model import build_model
import time
from src.metrics import Bleu, Meteor, Rouge
from pathlib import Path
import tqdm
from src.retriever import build_retriever

logger = logging.getLogger(__name__)

def retrieval_predict(model, data:Dataset, device:torch.device,
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
                               batch_size=batch_size, num_workers=num_workers)

    model.eval()

    # Place holders for scores.
    valid_scores = {"loss":float("nan"), "ppl":float("nan"), "bleu":float(0),"meteor":float(0), "rouge-l":float(0)}
    total_nseqs = 0
    total_ntokens = 0
    total_loss = 0
    all_outputs = []
    valid_sentences_scores = []
    valid_attention_scores = [] 
    all_batch_words = []
    hyp_scores = None
    attention_scores = None

    for batch_data in tqdm.tqdm(data_iter, desc="Validating"):
        batch_data.move2cuda(device)
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
                trg_truth = batch_data.trg_truth
                copy_param = dict()
                copy_param["source_maps"] = batch_data.src_maps
                copy_param["alignments"] = batch_data.alignments
                copy_param["src_vocabs"] = batch_data.src_vocabs
                blank_arr, fill_arr = collapse_copy_scores(model.trg_vocab, batch_data.src_vocabs)
                copy_param["blank_arr"] = blank_arr
                copy_param["fill_arr"] = fill_arr


                # batch_loss = model(return_type="loss", src_input=src_input, trg_input=trg_input,
                #    src_mask=src_mask, trg_mask=trg_mask, encoder_output = None, trg_truth=trg_truth, 
                #    copy_param=copy_param)
                
                # sum over multiple gpus.
                # if normalization = 'sum', keep the same.
                # batch_loss = batch_data.normalize(batch_loss, "sum")
                # if return_prob == "references":
                #     output = trg_truth
            
            # total_loss += batch_loss.item()
            total_ntokens += batch_data.ntokens
        
        # if return_prob == "ref", then no search need. Just look up the prob of the ground truth.
        if return_prob != "references":
            # run search as during inference to produce translations(summary)
            output, hyp_scores, attention_scores, batch_words = search(model=model, batch_data=batch_data,
                                                          beam_size=beam_size, beam_alpha=beam_alpha,
                                                          max_output_length=max_output_length, 
                                                          min_output_length=min_output_length, n_best=n_best,
                                                          return_attention=return_attention, return_prob=return_prob,
                                                          generate_unk=generate_unk, repetition_penalty=repetition_penalty,
                                                          no_repeat_ngram_size=no_repeat_ngram_size, copy_param=copy_param)
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
        all_batch_words.extend(batch_words if batch_words is not None else [])

    assert total_nseqs == len(data)
    # FIXME all_outputs is a list of np.ndarray
    assert len(all_outputs) == len(data) * n_best

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
    
    if model.copy is False:
        # decode ids back to str symbols (cut_off After eos, but eos itself is included. ) # NOTE eos is included.
        decoded_valid = model.trg_vocab.arrays_to_sentences(arrays=all_outputs, cut_at_eos=True)
    else:
        # copy mode
        decoded_valid = cut_off(all_batch_words, cut_at_eos=True, skip_pad=True)

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
                try:
                    valid_scores[eval_metric] = Meteor().compute_score(gts=references_dict, res=predictions_dict)[0]
                except:
                    logger.warning("meteor compute has something wrong!")
            elif eval_metric == "rouge-l":
                valid_scores[eval_metric] = Rouge().compute_score(gts=references_dict, res=predictions_dict)[0]
        eval_duration = time.time() - eval_metric_start_time
        eval_metrics_string = ", ".join([f"{eval_metric}:{valid_scores[eval_metric]:6.3f}" for eval_metric in 
                                          eval_metrics+["loss","ppl"]])

        logger.info("Evaluation result(%s) %s, evaluation time: %.4f[sec]", "Beam Search" if beam_size > 1 else "Greedy Search",
                     eval_metrics_string, eval_duration)
    else:
        logger.info("No data trg truth provided.")
    
    return (valid_scores, bleu_order, valid_references, valid_hypotheses, decoded_valid,
            valid_sentences_scores, valid_attention_scores)

def retrieval_test(cfg_file: str) -> None:
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

    train_data, dev_data, test_data, src_vocab, trg_vocab = load_data(data_cfg=cfg["data"])
    data_to_predict = {"dev": dev_data, "test":test_data}

    # build an transformer(encoder-decoder) model
    model = build_model(model_cfg=cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)

    # load model state from trained model
    pre_trained_model_path = cfg["retrieval_training"].get("pre_trained_model_path", None)
    use_cuda = cfg["retrieval_training"].get("use_cuda", False)
    device = torch.device("cuda" if use_cuda else "cpu")
    model_checkpoint = load_model_checkpoint(path=Path(pre_trained_model_path), device=device)

    # build an transformer(encoder-decoder) model
    model = build_model(model_cfg=cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)
    # FIXME train mangager has initialize from checkpoint
    model.load_state_dict(model_checkpoint["model_state"])

    retriever = build_retriever(cfg=cfg["retriever"])
    # load combiner from checkpoint for dynamic combiners
    model.retriever = retriever
    model.log_parameters_list()