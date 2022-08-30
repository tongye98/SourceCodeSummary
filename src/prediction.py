# coding: utf-8
"""
This module for generating prediction of a model.
"""
import logging 
import torch 
from torch.utils.data import Dataset
from typing import Dict
from helps import parse_test_arguments
import math
from datas import make_data_iter
from search import search
import time

logger = logging.getLogger(__name__)

def predict(model, data:Dataset, device:torch.device,
            n_gpu:int, compute_loss:bool=False, normalization:str="batch",
            num_workers:int=0, cfg:Dict=None):
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
        decoding_description = ("(Greedy decoding with " if beam_size < 2 else f"(Beam search with \
            beam_size={beam_size}, beam_alpha={beam_alpha}, n_best={n_best}")
        decoding_description += (f"min_output_length={min_output_length}, \
            max_output_length={max_output_length}, return_prob={return_prob}, \
            genereate_unk={generate_unk}, repetition_penalty={repetition_penalty}, \
            no_repeat_ngram_size={no_repeat_ngram_size})")
    logger.info("Predicting %d examples...%s", len(data), decoding_description)

    assert batch_size >= n_gpu, "batch size must be bigger than n_gpu"

    data_iter = make_data_iter(dataset=data, shuffle=False, batch_type=batch_type, batch_size=batch_size,
                               num_workers=num_workers, device=device)

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

        if compute_loss and batch_data.has_trg:
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
                total_tokens += batch_data.ntokens
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
    valid_references = data.trg

    if data.has_trg:
        valid_hyp_1best = (valid_hypotheses if n_best == 1 else [valid_hypotheses[i] for i in range(0, len(valid_hypotheses), n_best)])
        assert len(valid_hyp_1best) == len(valid_references)

        eval_metric_start_time = time.time()
        for eval_metric in eval_metrics:
            if eval_metric == "bleu":
                valid_scores[eval_metric] = 0
            elif eval_metric == "meteor":
                valid_scores[eval_metric] = 0
            elif eval_metric == "rouge-l":
                valid_scores[eval_metric] = 0
        eval_duration = time.time() - eval_metric_start_time
        eval_metrics_string = ", ".join([f"{eval_metric}:{valid_scores[eval_metric]:6.2f}" for eval_metric in 
                                          eval_metrics+["loss","ppl"] ])

        logger.info("Evaluation result(%s) %s, evaluation time: %.4f[sec]", "Beam Search" if beam_size > 1 else "Greedy Search",
                     eval_metrics_string, eval_duration)
    else:
        logger.info("No data trg truth provided.")
    
    return (valid_scores, valid_references, valid_hypotheses, decoded_valid,
            valid_sentences_scores, valid_attention_scores)