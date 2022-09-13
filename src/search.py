# coding: utf-8
"""
Search Module
"""
from tokenize import cookie_re
from typing import List, Tuple
from src.batch import Batch
import torch 
import torch.nn.functional as F
from src.helps import collapse_copy_scores, tile, tensor2sentence_copy
import numpy as np
from torch import Tensor 
from src.vocabulary import Vocabulary
import logging 

logger = logging.getLogger(__name__)

def search(model, batch_data: Batch, 
           beam_size: int, beam_alpha: float, 
           max_output_length: int, 
           min_output_length: int, n_best: int,
           return_attention: bool, return_prob: str, 
           generate_unk: bool, repetition_penalty: float, 
           no_repeat_ngram_size: float, copy_param=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get outputs and attention scores for a given batch.
    return:
        - stacked_output : hypotheses for batch
        - stacked_scores : log probability for batch
        - stacked_attention_scores: attention scores for batch
    """
    with torch.no_grad():
        src_input = batch_data.src
        src_mask = batch_data.src_mask
        encoder_output = model(return_type="encode", src_input=src_input, src_mask=src_mask)
        # encode_output [batch_size, src_len, model_dim]

        if beam_size < 2: # Greedy Strategy
            stacked_output, stacked_scores, stacked_attention_scores, batch_words = greedy_search(model, encoder_output, src_mask, max_output_length, min_output_length,
             generate_unk, return_attention, return_prob, repetition_penalty, no_repeat_ngram_size, copy_param)
        else:
            stacked_output, stacked_scores, stacked_attention_scores = beam_search(model, encoder_output, src_mask, max_output_length, min_output_length, 
             beam_size, beam_alpha, n_best, generate_unk, return_attention, return_prob, repetition_penalty, no_repeat_ngram_size)
        
        return stacked_output, stacked_scores, stacked_attention_scores, batch_words


def greedy_search(model, encoder_output, src_mask, max_output_length, min_output_length, 
                  generate_unk, return_attention, return_prob, repetition_penalty, no_repeat_ngram_size,
                  copy_param):
    """
    Transformer Greedy function.
    :param: model: Transformer Model
    :param: encoder_output: [batch_size, src_len, model_dim]
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
    if model.copy:
        generated_tokens_copy = encoder_output.new_full((batch_size,1), bos_index, dtype=torch.long, requires_grad=False)
        # generated_tokens_copy [batch_size, 1] generated_tokens id
        source_maps = copy_param["source_maps"]
        src_vocabs = copy_param["src_vocabs"]
        blank_arr = copy_param["blank_arr"]
        fill_arr = copy_param["fill_arr"]

    # Placeholder for scores
    generated_scores = generated_tokens.new_zeros((batch_size,1), dtype=torch.float) if return_prob == "hypotheses" else None
    # generated_scores [batch_size, 1]

    # Placeholder for attentions
    generated_attention_weight = generated_tokens.new_zeros((batch_size, 1, src_length), dtype=torch.float) if return_attention else None
    # generated_attention_weight [batch_size, 1, src_len]

    # a subsequent mask is intersected with this in decoder forward pass
    trg_mask = src_mask.new_ones((1, 1, 1))

    finished = src_mask.new_zeros(batch_size).byte() # [batch_size], uint8

    for step in range(max_output_length):
        with torch.no_grad():
            output, penultimate_representation, cross_attention_weight = model(return_type="decode", trg_input=generated_tokens, encoder_output=encoder_output,
                                                          src_mask=src_mask, trg_mask=trg_mask)
            # output [batch_size, trg_len, model_dim] -> [batch_size, step+1, model_dim]
            # cross_attention_weight [batch_size, trg_len, src_len] -> [batch_size, step+1, src_len]
            if model.copy is False:
                output = model.output_layer(output)
                # output [batch_size, step+1, trg_vocab_size]
                output = output[:, -1] 
                # output [batch_size, trg_vocab_size]
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
            else:
                fuse_score, attention_score = model.copy_attention_score(output, encoder_output, src_mask)
                prob = model.copy_generator(output, attention_score, source_maps)
                # prob [batch_size, step+1， trg_vocab_size + extra_words]
                output = prob[:, -1]
                # output [batch_size, trg_vocab_size + extra_words]
                if not generate_unk:
                    offset = model.trg_vocab_size
                    output[:, unk_index] = float("-inf")
                    output[:, unk_index + offset] = float("-inf")
                if step < min_output_length:
                    output[:, eos_index] = float("-inf")
                if compute_softmax:
                    output = F.log_softmax(output, dim=-1)
                
                for sentence_id in range(output.size(0)):
                    blank = torch.LongTensor(blank_arr[sentence_id]).cuda(non_blocking=True)
                    fill = torch.LongTensor(fill_arr[sentence_id]).cuda(non_blocking=True)
                    output[sentence_id].index_add_(0, fill, output[sentence_id].index_select(0, blank))
                    output[sentence_id].index_fill_(0, blank, 1e-10)

            # take the most likely token
            prob, next_words = torch.max(output, dim=-1)
            # prob [batch_size]
            # next_words [batch_size]
            next_words = next_words.data

            if model.copy is False:
                generated_tokens = torch.cat([generated_tokens, next_words.unsqueeze(-1)], dim=-1) # [batch_size, step+2]
            else:   
                # FIXME if next word id > trg_vocab_size? 
                # first step:  generate tokens with src vocab
                generated_tokens_copy = torch.cat([generated_tokens_copy, next_words.unsqueeze(-1)], dim=-1) # [batch_size, step+2]
                for id in range(next_words.size(0)):
                    if next_words[id].item() > len(model.trg_vocab):
                        next_words[id] = 0   # for those generated src vocab tokens(not in trg vocab), set to unk 
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
    
    batch_words = None
    if model.copy:
        generated_tokens_copy = generated_tokens_copy[:, 1:]
        batch_words = tensor2sentence_copy(generated_tokens_copy, model.trg_vocab, src_vocabs)

    # Remove bos-symbol
    stacked_output = generated_tokens[:, 1:].detach().cpu().numpy()
    stacked_scores = generated_scores[:, 1:].detach().cpu().numpy() if return_prob == "hypotheses" else None
    stacked_attention = generated_attention_weight[:, 1:, :].detach().cpu().numpy() if return_attention else None

    return stacked_output, stacked_scores, stacked_attention, batch_words



def beam_search(model, encoder_output, src_mask, max_output_length, min_output_length, beam_size, beam_alpha,
                n_best, generate_unk, return_attention, return_prob, repetition_penalty, no_repeat_ngram_size):
    """
    Transformer Beam Search function.
    In each decoding step, find the k most likely partial hypotheses.
    Inspired by OpenNMT-py, adapted for Transformer.
    :param: model: Transformer Model
    :param: encoder_output: [batch_size, src_len, model_dim]
    :param: src_mask: [batch_size, 1, src_len] # src_len is padded src length
    return
        - final_output [batch_size*n_best, hyp_len]
        - scores
        - attention: None 
    """
    assert beam_size > 0, "Beam size must be > 0."
    assert n_best <= beam_size, f"Can only return {beam_size} best hypotheses."

    unk_index = model.unk_index
    pad_index = model.pad_index
    bos_index = model.bos_index 
    eos_index = model.eos_index
    batch_size = src_mask.size(0)

    trg_vocab_size = model.decoder.output_size
    trg_mask = None 
    device = encoder_output.device

    encoder_output = tile(encoder_output.contiguous(), beam_size, dim=0)
    # encoder_output [batch_size*beam_size, src_len, model_dim] i.e. [a,a,a,b,b,b]
    src_mask = tile(src_mask, beam_size, dim=0)
    # src_mask [batch_size*beam_size, 1, src_len]

    trg_mask = src_mask.new_ones((1,1,1))

    batch_offset = torch.arange(batch_size, dtype=torch.long, device=device) # [0,1,2,... batch_size-1]
    beam_offset = torch.arange(0, batch_size*beam_size, step=beam_size, dtype=torch.long, device=device)
    # beam_offset [0,5,10,15,....] i.e. beam_size=5

    # keep track of the top beam size hypotheses to expand for each element
    # in the batch to be futher decoded (that are still "alive")
    alive_sentences = torch.full((batch_size*beam_size, 1), bos_index, dtype=torch.long, device=device)
    # alive_sentences [batch_size*beam_size, hyp_len] now is [batch_size*beam_size, 1]

    top_k_log_probs = torch.zeros(batch_size, beam_size, device=device)
    top_k_log_probs[:, 1:] = float("-inf")

    # Structure that holds finished hypotheses.
    hypotheses = [[] for _ in range(batch_size)]

    results = {"predictions": [[] for _ in range(batch_size)], 
                "scores": [[] for _ in range(batch_size)] }

    # Indicator if the generation is finished.
    is_finished = torch.full((batch_size, beam_size), False, dtype=torch.bool, device=device)

    for step in range(max_output_length):
        # feed the complete predicted sentences so far.
        decoder_input = alive_sentences
        with torch.no_grad():
            output, input, cross_attention_weight = model(return_type="decode", trg_input=decoder_input, encoder_output=encoder_output,
                                                          src_mask=src_mask, trg_mask=trg_mask)
            # output: after output layer [batch_size*beam_size, trg_len, vocab_size] -> [batch_size*beam_size, step+1, vocab_size]
            # input: before output layer [batch_size*beam_size, trg_len, model_dim] -> [batch_size*beam_size, step+1, model_dim]
            # cross_attention_weight [batch_size*beam_size, trg_len, src_len] -> [batch_size*beam_size, step+1, src_len]

            # for the transformer we made predictions for all time steps up to this point,
            # so we only want to know about the last time step.
            output = output[:, -1] # output [batch_size*beam_size, vocab_size]

        # compute log probability distribution over trg vocab
        log_probs = F.log_softmax(output, dim=-1)
        # log_probs [batch_size*beam_size, vocab_size]

        if not generate_unk:
            log_probs[:, unk_index] = float("-inf")
        # don't genereate EOS symbol until we reach min_output_length
        if step < min_output_length:
            log_probs[:, eos_index] = float("-inf")
        
        #TODO
        # no_repeat_ngram_size
        # repetition_penalty

        # multiply probs by the beam probability (means add log_probs after log operation)
        log_probs += top_k_log_probs.view(-1).unsqueeze(1)
        current_scores = log_probs.clone()

        # compute length penalty
        if beam_alpha > 0:
            length_penalty = ((5.0 + (step+1)) / 6.0)**beam_alpha
            current_scores /= length_penalty
        
        # flatten log_probs into a list of possibilities
        current_scores = current_scores.reshape(-1, beam_size*trg_vocab_size)
        # current_scores [batch_size, beam_size*vocab_size]

        # pick currently best top k hypotheses
        topk_scores, topk_ids =current_scores.topk(beam_size, dim=-1)
        # topk_scores [batch_size, beam_size]
        # topk_ids [batch_size, beam_size]

        if beam_alpha > 0:
            top_k_log_probs = topk_scores * length_penalty
        else: 
            top_k_log_probs = topk_scores.clone()
        
        # Reconstruct beam origin and true word ids from flatten order
        topk_beam_index = topk_ids.div(trg_vocab_size, rounding_mode="floor")
        # topk_beam_index [batch_size, beam_size]
        topk_ids = topk_ids.fmod(trg_vocab_size) # true word ids
        # topk_ids [batch_size, beam_size]

        # map topk_beam_index to batch_index in the flat representation
        batch_index = topk_beam_index + beam_offset[:topk_ids.size(0)].unsqueeze(1)
        # batch_index [batch_size, beam_size]
        select_indices = batch_index.view(-1)
        # select_indices [batch_size*beam_size]: the number of seleced index in the batch.

        # append latest prediction
        alive_sentences = torch.cat([alive_sentences.index_select(0, select_indices), topk_ids.view(-1, 1)], dim=-1)
        # alive_sentences [batch_size*beam_size, hyp_len]

        is_finished = topk_ids.eq(eos_index) | is_finished | topk_scores.eq(-np.inf)
        # is_finished [batch_size, beam_size]
        if step + 1 == max_output_length:
            is_finished.fill_(True)
        
        # end condition is whether all beam candidates in each example are finished.
        end_condition = is_finished.all(dim=-1)
        # end_condition [batch_size]

        # save finished hypotheses
        if is_finished.any():
            predictions = alive_sentences.view(-1, beam_size, alive_sentences.size(-1))
            # predictions [batch_size, beam_size, hyp_len]

            for sentence_idx in range(is_finished.size(0)): # look over sentences
                b = batch_offset[sentence_idx].item() # index of that example in the batch
                if end_condition[sentence_idx]:
                    is_finished[sentence_idx].fill_(True)
                
                finished_hyp = is_finished[sentence_idx].nonzero(as_tuple=False).view(-1)
                for sentence_beam_idx in finished_hyp: # look over finished beam candidates
                    number_eos = (predictions[sentence_idx, sentence_beam_idx, 1:] == eos_index).count_nonzero().item()
                    if number_eos > 1: # prediction should have already been added to the hypotheses
                        continue
                    elif (number_eos == 0 and step+1 == max_output_length) or (number_eos == 1 and predictions[sentence_idx, sentence_beam_idx, -1] == eos_index):
                        hypotheses[b].append((topk_scores[sentence_idx, sentence_beam_idx], predictions[sentence_idx, sentence_beam_idx,1:]))

                # if all n best candidates of the i-the example reached the end, save them
                if end_condition[sentence_idx]:
                    best_hyp = sorted(hypotheses[b], key=lambda x:x[0], reverse=True)
                    for n, (score, pred) in enumerate(best_hyp):
                        if n >= n_best:
                            break 
                        if len(pred) < max_output_length:
                            assert pred[-1] == eos_index, "Add a candidate which doesn't end with eos."
                        
                        results['scores'][b].append(score)
                        results['predictions'][b].append(pred)
            
            # batch indices of the examples which contain unfinished candidates.
            unfinished = end_condition.eq(False).nonzero(as_tuple=False).view(-1)
            # unfinished [batch_size]
            if len(unfinished) == 0:
                break
            
            # remove finished examples for the next steps.
            # shape [remaining_batch_size, beam_size]
            batch_index = batch_index.index_select(0, unfinished)
            top_k_log_probs = top_k_log_probs.index_select(0, unfinished)
            is_finished = is_finished.index_select(0, unfinished)
            batch_offset = batch_offset.index_select(0, unfinished)

            alive_sentences = predictions.index_select(0, unfinished).view(-1, alive_sentences.size(-1))

        # Reorder indices, outputs and masks
        select_indices = batch_index.view(-1)
        encoder_output = encoder_output.index_select(0, select_indices)
        src_mask = src_mask.index_select(0, select_indices)

    def pad_and_stack_hyps(hyps: List[np.ndarray]):
        filled = (np.ones((len(hyps), max([h.shape[0]  for h in hyps])), dtype=int) * pad_index)
        for j, h in enumerate(hyps):
            for k, i in enumerate(h):
                filled[j, k] = i
        return filled

    # from results to stacked output
    # final_outputs [batch_size*n_best, hyp_len]
    predictions_list = [u.cpu().numpy() for r in results["predictions"] for u in r]
    final_outputs = pad_and_stack_hyps(predictions_list)
    scores = (np.array([[u.item()] for r in results['scores'] for u in r]) if return_prob else None)

    return final_outputs, scores, None