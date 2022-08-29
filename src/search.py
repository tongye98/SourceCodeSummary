# coding: utf-8
"""
Search Module
"""
from batch import Batch
import torch 

def search(model, batch_data: Batch, 
           beam_size: int, beam_alpha: float, 
           max_output_length: int, n_best: int,
           return_attention: bool, return_prob: str, 
           generate_unk: bool, repetition_penalty: float, 
           no_repeat_ngram_size: float):
    """
    Get outputs and attention scores for a given batch.
    return:
        - stacked_output
        - stacked_scores
        - stacked_attention_scores
    """
    with torch.no_grad():
        src_input = batch_data.src
        src_mask = batch_data.src_mask
        encode_output = model(return_type="encode", src_input=src_input, src_mask=src_mask)
        # encode_output [batch_size, src_len, model_dim]

        if beam_size < 2: # Greedy Strategy
            greedy()
        else:
            beam_search()
        
        return None 


def greedy():
    return None

def beam_search():
    return None