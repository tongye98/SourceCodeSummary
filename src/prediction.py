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

logger = logging.getLogger(__name__)

def predict(model, data:Dataset, device:torch.device,
            n_gpu:int, compute_loss:bool=False, normalization:str="batch",
            num_workers:int=0, test_cfg:Dict=None):
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
     return_prob, generate_unk, repetition_penalty, no_repeat_ngram_size) = parse_test_arguments(test_cfg)

    # FIXME what is return_prob
    if return_prob == "ref":
        decoding_description = ""
    else:
        decoding_description = ("(Greedy decoding with " if beam_size<2 else f"(Beam search with \
            beam_size={beam_size}, beam_alpha={beam_alpha}, n_best={n_best}")
        decoding_description += (f"min_output_length={min_output_length}, \
            max_output_length={max_output_length}, return_prob={return_prob}, \
            genereate_unk={generate_unk}, repetition_penalty={repetition_penalty}, \
            no_repeat_ngram_size={no_repeat_ngram_size})")
    logger.info("Predicting %d examples...%s", len(data), decoding_description)

    assert batch_size >= n_gpu, "batch size must be bigger than n_gpu"

    data_iter = make_data_iter()

    model.eval()

    # Place holders for scores.
    valid_scores = {"loss":float("nan"), "ppl":float("nan"), "bleu":float(0),"meteor":float(0), "rouge-l":float(0)}

    for batch_data in data_iter:

        if compute_loss and batch_data.has_trg:
            assert model.loss_function is not None
            # don't track gradients during validation 
            with torch.no_grad():
                batch_loss, log_probs = model(return_type="loss", src_input=None, trg_input=None,
                   src_mask=None, trg_mask=None, encoder_output = None)
                
                # sum over multiple gpus.
                # TODO

                if return_prob == "ref":
                    output = batch_data.trg
            
            total_loss += batch_loss.item()
            total_ntokens += batch_data.ntokens
        
        # if return_prob == "ref", then no search need. 
        # just look up the prob of the ground truth.

        if return_prob != "ref":
            # run search as during inference to produce translations(summary)
            output, hyp_scores, attention_scores = search()
        
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