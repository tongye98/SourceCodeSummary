# coding: utf-8
"""
Search Module
"""
from batch import Batch
import torch 
import torch.nn.functional as F

def search(model, batch_data: Batch, 
           beam_size: int, beam_alpha: float, 
           max_output_length: int, 
           min_output_length: int, n_best: int,
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
        encoder_output = model(return_type="encode", src_input=src_input, src_mask=src_mask)
        # encode_output [batch_size, src_len, model_dim]

        if beam_size < 2: # Greedy Strategy
            greedy_search(model, encoder_output, src_mask, max_output_length, min_output_length, generate_unk, 
                          return_attention, return_prob, repetition_penalty, no_repeat_ngram_size)
        else:
            beam_search()
        
        return None 


def greedy_search(model, encoder_output, src_mask, max_output_length, min_output_length, 
                  generate_unk, return_attention, return_prob, repetition_penalty, no_repeat_ngram_size):
    """
    Transformer Greedy function.
    :param: model: Transformer Model
    :param: encode_output: [batch_size, src_len, model_dim]
    :param: src_mask: [batch_size, 1, src_len] # src_len is padded src length
    return
        - stacked_output [batch_size, steps/max_output_length]
        - stacked_scores [batch_size, steps/max_output_length] # log_softmax token probability
        - stacked_attention [batch_size, steps/max_output_length, src_len]
    """
    unk_index = model.unk_index
    pad_index = model.pad_index
    bos_index = model.bos_index
    eos_index = model.eos_index 
    # FIXME 
    compute_softmax = return_prob == "hypotheses"

    batch_size, _, src_length = src_mask.size()

    # start with BOS-symbol for each sentence in the batch
    generated_tokens = encoder_output.new_full((batch_size,1), bos_index, dtype=torch.long, requires_grad=False)
    # generated_tokens [batch_size, 1] generated_tokens id

    # Placeholder for scores
    generated_scores = generated_tokens.new_zeros((batch_size,1), dtype=torch.float) if return_prob == "hypotheses" else None
    # generated_scores [batch_size, 1]

    # Placeholder for attentions
    generated_attention_weight = generated_tokens.new_zeros((batch_size, 1, src_length), dtype=torch.float) if return_attention else None
    # generated_attention_weight [batch_size, 1, src_len]

    # a subsequent mask is intersected with this in decoder forward pass
    # FIXME for torch.nn.DataParallel
    trg_mask = src_mask.new_ones((1, 1, 1))

    finished = src_mask.new_zeros(batch_size).byte() # [batch_size], uint8

    for step in range(max_output_length):
        with torch.no_grad():
            output, input, cross_attention_weight = model(return_type="decode", trg_input=generated_tokens, encoder_output=encoder_output,
                                                          src_mask=src_mask, trg_mask=trg_mask)
            # output: after output layer [batch_size, trg_len, vocab_size] -> [batch_size, step+1, vocab_size]
            # input: before output layer [batch_size, trg_len, model_dim] -> [batch_size, step+1, model_dim]
            # cross_attention_weight [batch_size, trg_len, src_len] -> [batch_size, step+1, src_len]
            output = output[:, -1] # output [batch_size, vocab_size]
            if not generate_unk:
                output[:, unk_index] = float("-inf")
            # Don't generate EOS until we reached min_output_length
            if step < min_output_length:
                output[:, eos_index] = float("-inf")
            
            if compute_softmax:
                output = F.log_softmax(output, dim=-1)
                #TODO
                # no_repeat_ngram_size
                # penalize_repetition
            
            # take the most likely token
            prob, next_words = torch.max(output, dim=-1)
            # prob [batch_size]
            # next_words [batch_size]
            next_words = next_words.data
            generated_tokens = torch.cat([generated_tokens, next_words.unsqueeze(-1)], dim=-1) # [batch_size, step+2]

            if return_prob == "hypotheses":
                prob = prob.data
                generated_scores = torch.cat([generated_scores, prob.unsqueeze(-1)], dim=-1) # [batch_size, step+2]
            if return_attention is True:
                assert cross_attention_weight is not None 
                cross_attention = cross_attention_weight.data[:, -1, :].unsqueeze(1) # [batch_size, 1, src_len]
                generated_attention_weight = torch.cat([generated_attention_weight, cross_attention], dim=1) # [batch_size, step+2, src_len]
        
        # check if previous symbol is <eos>
        is_eos = torch.eq(next_words, eos_index)
        finished += is_eos
        if (finished >= 1).sum() == batch_size:
            break
    
    # Remove bos-symbol
    stacked_output = generated_tokens[:, 1:].detach().cpu().numpy()
    stacked_scores = generated_scores[:, 1:].detach().cpu().numpy() if return_prob == "hypotheses" else None
    stacked_attention = generated_attention_weight[:, 1:, :].detach().cpu().numpy() if return_attention else None

    return stacked_output, stacked_scores, stacked_attention


def beam_search():
    """
    Transformer Beam Search function.
    """
    return None