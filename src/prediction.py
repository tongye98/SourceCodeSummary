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
    total_tokens = 0
    all_outputs = []
    valid_attention_scores = [] 

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
        valid_attention_scores.extend(attention_scores)

    if compute_loss:
        valid_scores["loss"] = total_loss 
        valid_scores["ppl"] = math.exp(total_loss / total_ntokens)
    
    decoded_valid = model.trg_vocab.arrays_to_sentences(arrays=all_outputs, cut_at_eos=True)

    valid_hyp = [data.tokenizer[data.trg_lang].post_process(s, generate_unk=generate_unk) for s in decoded_valid]

    valid_ref = data.trg

    if data.has_trg:
        valid_hyp_1best = None 
        for eval_metric in eval_metrics:
            if eval_metric == "bleu":
                pass
            if eval_metric == "meteor":
                pass 
            if eval_metric == "rouge-l":
                pass
    
    return (valid_scores, valid_ref, valid_hyp, decoded_valid,
            valid_sequence_scores, valid_attention_scores,)