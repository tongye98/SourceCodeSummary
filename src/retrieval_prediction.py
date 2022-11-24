# coding: utf-8
"""
This module for generating prediction of a model.
Greedy search and Beam search.
"""
import logging 
import torch 
import math
import time
import tqdm
import numpy as np
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
from src.helps import  tile, parse_test_arguments
from src.datas import Batch, make_data_iter
from src.metrics import eval_accuracies

logger = logging.getLogger(__name__)

def predict(model, data:Dataset, device:torch.device, compute_loss:bool=False, 
            normalization:str="batch", num_workers:int=0, test_cfg:Dict=None, use_code_representation: bool=False):
    """
    Generate outputs(translations/summary) for the given data.
    If 'compute_loss' is True, also computes the loss.
    return:
        - valid_scores (loss, ppl, bleu, meteor, rouge-l)
        - valid_references
        - valid_hypotheses
        - valid_sentences_scores
        - valid_attention_scores
    """
    (batch_size, batch_type, max_output_length, min_output_length,
     eval_metrics, beam_size, beam_alpha, n_best, return_attention, 
     return_prob, generate_unk, repetition_penalty) = parse_test_arguments(test_cfg)

    decoding_description = ("(Greedy decoding with " if beam_size < 2 else f"(Beam search with "
                            f"beam_size={beam_size}, beam_alpha={beam_alpha}, n_best={n_best}")
    decoding_description += (f"min_output_length={min_output_length}, "
                                f"max_output_length={max_output_length}, return_prob={return_prob}, "
                                f"genereate_unk={generate_unk}, batch_size={batch_size})")
    logger.info("Predicting %d examples...%s", len(data), decoding_description)

    data_iter = make_data_iter(dataset=data, sampler_seed=None, shuffle=False, batch_type=batch_type,
                               batch_size=batch_size, num_workers=num_workers)

    model.eval()

    valid_scores = {"loss":float("nan"), "ppl":float("nan"), "bleu":float(0), "meteor":float(0), "rouge-l":float(0)}
    total_nseqs = 0
    total_ntokens = 0
    total_loss = 0

    all_outputs = []
    valid_sentences_scores = []
    valid_attention_scores = [] 
    all_batch_words = []
    hyp_scores = None
    attention_scores = None

    # for retrieval analysis
    all_hits = 0.
    all_first_hits = 0.
    all_token_numbers = 0.
    help_token_num = 0.
    model_true_mix_true_num = 0.
    model_true_mix_false_num = 0.
    model_false_mix_true_num = 0.
    model_false_mix_false_num = 0.

    full_distance_list = list()
    inference_start_time = time.time()
    for batch_data in tqdm.tqdm(data_iter, desc="Validating"):
        batch_data.move2cuda(device)
        total_nseqs += batch_data.nseqs 

        # if compute_loss: 
        #     # don't track gradients during validation 
        #     with torch.no_grad():
        #         src_input = batch_data.src
        #         trg_input = batch_data.trg_input
        #         src_mask = batch_data.src_mask
        #         trg_mask = batch_data.trg_mask
        #         trg_truth = batch_data.trg_truth

        #         batch_loss, retrieval_analysis, help_analysis = model(return_type="retrieval_loss", src_input=src_input, trg_input=trg_input,
        #         src_mask=src_mask, trg_mask=trg_mask, encoder_output=None, trg_truth=trg_truth)
                
        #         batch_loss = batch_data.normalize(batch_loss, "sum")

        #     total_loss += batch_loss.item()
        #     total_ntokens += batch_data.ntokens

        # all_hits += retrieval_analysis["hits"]
        # all_first_hits += retrieval_analysis["hits_first_place"]
        # all_token_numbers += retrieval_analysis["token_numbers"]

        # help_token_num +=help_analysis["token_num"]
        # model_true_mix_true_num += help_analysis["model_true_mix_true_num"]
        # model_true_mix_false_num += help_analysis["model_true_mix_false_num"]
        # model_false_mix_true_num += help_analysis["model_false_mix_true_num"]
        # model_false_mix_false_num += help_analysis["model_false_mix_false_num"]

        # run search as during inference to produce translations (summary).
        output, hyp_scores, attention_scores, batch_distance_list = search(model=model, batch_data=batch_data,
                beam_size=beam_size, beam_alpha=beam_alpha, max_output_length=max_output_length, 
                min_output_length=min_output_length, n_best=n_best, return_attention=return_attention,
                return_prob=return_prob, generate_unk=generate_unk, repetition_penalty=repetition_penalty,
                use_code_representation=use_code_representation)

        all_outputs.extend(output)
        valid_sentences_scores.extend(hyp_scores if hyp_scores is not None else [])
        valid_attention_scores.extend(attention_scores if attention_scores is not None else [])
        full_distance_list.extend(batch_distance_list)
    inference_end_time = time.time()
    logger.info("Inference Time Cost = {}".format(inference_end_time - inference_start_time))

    # logger.info("all_hits = {}, all_token_numbers = {}, hit_accuracy = {}".format(all_hits, all_token_numbers, all_hits/all_token_numbers))
    # logger.info("first_hits={}, all_token_numbers = {}, first hit_accuracy = {}".format(all_first_hits, all_token_numbers,all_first_hits/all_token_numbers))
    # logger.info("model_true_mix_true_num = {} help_token_num = {}, ratio = {}".format(model_true_mix_true_num, help_token_num, model_true_mix_true_num/help_token_num))
    # logger.info("model_true_mix_false_num = {} help_token_num = {}, ratio = {}".format(model_true_mix_false_num, help_token_num, model_true_mix_false_num/help_token_num))
    # logger.info("model_false_mix_true_num = {} help_token_num = {}, ratio = {}".format(model_false_mix_true_num, help_token_num, model_false_mix_true_num/help_token_num))
    # logger.info("model_false_mix_false_num = {} help_token_num = {}, ratio = {}".format(model_false_mix_false_num, help_token_num, model_false_mix_false_num/help_token_num))

    assert total_nseqs == len(data)
    assert len(all_outputs) == len(data) * n_best # NOTE all_outputs is a list of np.ndarray

    if compute_loss:
        if normalization == "batch":
            normalizer = total_nseqs
        elif normalization == "tokens":
            normalizer = total_ntokens
        elif normalization == "none":
            normalizer = 1
        valid_scores["loss"] = total_loss / normalizer
        valid_scores["ppl"] = math.exp(total_loss / total_ntokens)
    
    decoded_valid = model.trg_vocab.arrays_to_sentences(arrays=all_outputs, cut_at_eos=True)

    # retrieve detokenized hypotheses and references
    valid_hypotheses = [data.tokenizer[data.trg_language].post_process(sentence, generate_unk=generate_unk) for sentence in decoded_valid]
    # valid_hypotheses -> list of strings
    valid_references = data.trg

    valid_hyp_1best = (valid_hypotheses if n_best == 1 else [valid_hypotheses[i] for i in range(0, len(valid_hypotheses), n_best)])
    assert len(valid_hyp_1best) == len(valid_references)

    predictions_dict = {k: [v.strip()] for k,v in enumerate(valid_hyp_1best)}
    references_dict = {k: [v.strip()] for k,v in enumerate(valid_references)}
    # 0: ['partitions a list of suite from a interval .']

    eval_metric_start_time = time.time()
    # for eval_metric in eval_metrics:
    #     if eval_metric == "bleu":  # geometric mean of bleu scores
    #         valid_scores[eval_metric], bleu_order = Bleu().corpus_bleu(hypotheses=predictions_dict, references=references_dict)
    #     elif eval_metric == "meteor":
    #         valid_scores[eval_metric] = 0
    #     elif eval_metric == "rouge-l":
    #         valid_scores[eval_metric] = Rouge().compute_score(gts=references_dict, res=predictions_dict)[0]
    bleu, rouge_l, meteor = eval_accuracies(hypotheses=predictions_dict, references=references_dict)
    eval_duration = time.time() - eval_metric_start_time
    # eval_metrics_string = ", ".join([f"{eval_metric}:{valid_scores[eval_metric]:6.3f}" for eval_metric in 
    #                                     eval_metrics+["loss","ppl"]])
    eval_metrics_string = "bleu = {}, rouge_l = {}, meteor = {}".format(bleu, rouge_l, meteor)
    logger.info("Evaluation result({}) {}, evaluation time: {:.2f}[sec]".format("Beam Search" if beam_size > 1 else "Greedy Search",
                    eval_metrics_string, eval_duration))

    return (valid_scores, valid_references, valid_hypotheses, valid_sentences_scores, valid_attention_scores)

def search(model, batch_data: Batch, beam_size: int, beam_alpha: float, max_output_length: int, 
           min_output_length: int, n_best: int, return_attention: bool, return_prob: str, 
           generate_unk: bool, repetition_penalty: float, use_code_representation: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        src_input = batch_data.src
        src_mask = batch_data.src_mask
        encoder_output = model(return_type="encode", src_input=src_input, src_mask=src_mask)
        if beam_size < 2:   # Greedy Search
            stacked_output, stacked_scores, stacked_attention_scores, batch_distance_list= greedy_search(model, encoder_output, src_mask, 
            max_output_length, min_output_length, generate_unk, return_attention, return_prob, use_code_representation)
        else:               # Beam Search
            stacked_output, stacked_scores, stacked_attention_scores = beam_search(model, encoder_output, src_mask, 
            max_output_length, min_output_length, beam_size, beam_alpha, n_best, generate_unk, 
            return_attention, return_prob, repetition_penalty, use_code_representation)
        
        return stacked_output, stacked_scores, stacked_attention_scores, [0,1]

def greedy_search(model, encoder_output, src_mask, max_output_length, min_output_length, 
                  generate_unk, return_attention, return_prob, use_code_representation):
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
    trg_mask = src_mask.new_ones((1, 1, 1))

    finished = src_mask.new_zeros(batch_size).byte() # [batch_size], uint8
    
    batch_encoder_output_list = []
    for sentence_encoder_output, sentence_src_mask in zip(encoder_output, src_mask):
        # sentence_encoder_output [src_len, model_dim]
        # sentence_src_mask [1, src_len]
        sentence_encoder_output_select = sentence_encoder_output[sentence_src_mask.squeeze(0)]
        # sentnce encoder output select [src_len_selected, model_dim]
        sentence_encode_concat = torch.mean(sentence_encoder_output_select, dim=0, keepdim=True)
        # sentence encode concat [1, model_dim]
        batch_encoder_output_list.append(sentence_encode_concat.unsqueeze(0))
    encode_concat = torch.cat(batch_encoder_output_list, dim=0)
    # encode_concat [batch_size, 1, model_dim]

    batch_distances_list = []
    for step in range(max_output_length):
        with torch.no_grad():
            output, penultimate_representation, cross_attention_weight = model(return_type="decode", trg_input=generated_tokens, 
                                                            encoder_output=encoder_output, src_mask=src_mask, trg_mask=trg_mask)
            # output  [batch_size, step+1, model_dim]
            # penultimate_representation [batch_size, step+1, model_dim]
            # cross_attention_weight [batch_size, step+1, src_len]
            output = model.output_layer(output)
            # output [batch_size, step+1, trg_vocab_size]
            output = output[:, -1].unsqueeze(1)
            # output [batch_size, 1, trg_vocab_size]
            penultimate_representation = penultimate_representation[:, -1].unsqueeze(1)
            # penultimate_representation [batch_size, 1, model_dim]

            if use_code_representation:
                hidden = torch.cat((encode_concat, penultimate_representation), dim=-1)
            else:
                hidden = penultimate_representation

            log_probs, analysis = model.retriever(hidden, output)
            # log_probs [batch_size, 1, vocab_size]
            log_probs = log_probs.squeeze(1)

            # distances = analysis["distances"] # distances torch.tensor on gpu
            # batch_distances_list.append(distances)

            if not generate_unk:
                log_probs[:, unk_index] = float("-inf")
            if step < min_output_length:
                log_probs[:, eos_index] = float("-inf")

        # take the most likely token
        prob, next_words = torch.max(log_probs, dim=-1)
        # prob [batch_size]
        # next_words [batch_size]

        generated_tokens = torch.cat([generated_tokens, next_words.unsqueeze(-1)], dim=-1) # [batch_size, step+2]
        generated_scores = torch.cat([generated_scores, prob.unsqueeze(-1)], dim=-1) # [batch_size, step+2]

        if return_attention is True:
            cross_attention = cross_attention_weight.data[:, -1, :].unsqueeze(1) # [batch_size, 1, src_len]
            generated_attention_weight = torch.cat([generated_attention_weight, cross_attention], dim=1) # [batch_size, step+2, src_len]
    
        # check if previous symbol is <eos>
        is_eos = torch.eq(next_words, eos_index)
        finished += is_eos
        if (finished >= 1).sum() == batch_size:
            break

    # batch_distance = torch.cat(distances_list)

    # Remove bos-symbol
    stacked_output = generated_tokens[:, 1:].detach().cpu().numpy()
    stacked_scores = generated_scores[:, 1:].detach().cpu().numpy() if return_prob == "hypotheses" else None
    stacked_attention = generated_attention_weight[:, 1:, :].detach().cpu().numpy() if return_attention else None

    return stacked_output, stacked_scores, stacked_attention, batch_distances_list

def beam_search(model, encoder_output, src_mask, max_output_length, min_output_length, beam_size, beam_alpha,
                n_best, generate_unk, return_attention, return_prob, repetition_penalty,use_code_representation):
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

    trg_vocab_size = model.trg_vocab_size
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
            output, penultimate_representation, cross_attention_weight = model(return_type="decode", trg_input=decoder_input, 
                                                            encoder_output=encoder_output, src_mask=src_mask, trg_mask=trg_mask)
            output = model.output_layer(output)
            # output  [batch_size*beam_size, step+1, vocab_size]
            # penultimate_representation [batch_size*beam_size, step+1, model_dim]
            # cross_attention_weight  [batch_size*beam_size, step+1, src_len]

            # for the transformer we made predictions for all time steps up to this point, so we only want to know about the last time step.
            # output = output[:, -1] # output [batch_size*beam_size, vocab_size]
            output = output[:, -1].unsqueeze(1) # [batch_size*beam_size, 1, vocab_size]
            cross_attention_weight = cross_attention_weight[:, -1].unsqueeze(1)
            penultimate_representation = penultimate_representation[:, -1].unsqueeze(1) # [batch_size*beam_size, 1, model_dim]

            # encode_output (selected, src_len, model_dim)
            # src_mask (seleced, 1, src_len)
            concat_list = []
            for sentence_encode_output, sentence_src_mask, sentence_cross_attention_weight in zip(encoder_output, src_mask, cross_attention_weight):
                #sentence_encode_output [src_len, model_dim]
                #sentence src_mask [1, src_len]
                #sentence cross attention weight [1, src_len]
                # selected_sentence_src_output = sentence_encode_output[sentence_src_mask.squeeze(0)]
                # mean = torch.mean(selected_sentence_src_output, dim=0, keepdim=True)
                # mean [1, model_dim]
                # concat_list.append(mean.unsqueeze(0))
                attention_mean = torch.matmul(sentence_cross_attention_weight, sentence_encode_output)
                concat_list.append(attention_mean.unsqueeze(0))
            encode_concat = torch.cat(concat_list, dim=0)
            # encode_concat [selected, 1, model_dim]
            # logger.warning("encode concat shape = {}".format(encode_concat.shape))
            # logger.warning("penultimate representation = {}".format(penultimate_representation.shape))
            if use_code_representation:
                hidden = torch.cat((encode_concat, penultimate_representation), dim=-1)
            else:
                hidden = penultimate_representation

            log_probs, _ = model.retriever(hidden, output)
            # log_probs, _ = model.retriever(hidden, output)
            log_probs = log_probs.squeeze(1)

        # compute log probability distribution over trg vocab
        # log_probs = F.log_softmax(output, dim=-1)
        # log_probs [batch_size*beam_size, vocab_size]

        if not generate_unk:
            log_probs[:, unk_index] = float("-inf")
        if step < min_output_length:
            log_probs[:, eos_index] = float("-inf")

        if repetition_penalty > 1.0:
            log_probs = penalize_repetition(alive_sentences, log_probs,
            repetition_penalty, exclude_tokens=[bos_index, eos_index, unk_index, pad_index])

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


def penalize_repetition(tokens: Tensor,
                        scores: Tensor,
                        penalty: float,
                        exclude_tokens: List[int] = None) -> Tensor:
    """
    Reduce probability of the given tokens.
    Taken from Huggingface's RepetitionPenaltyLogitsProcessor.

    :param tokens: token ids to penalize
    :param scores: log probabilities of the next token to generate
    :param penalty: penalty value, bigger value implies less probability
    :param exclude_tokens: list of token ids to exclude from penalizing
    """
    scores_before = scores if exclude_tokens else None
    score = torch.gather(scores, 1, tokens)

    # if score < 0 then repetition penalty has to be multiplied
    # to reduce the previous token probability
    score = torch.where(score < 0, score * penalty, score / penalty)

    scores.scatter_(1, tokens, score)

    # exclude special tokens
    if exclude_tokens:
        for token in exclude_tokens:
            # pylint: disable=unsubscriptable-object
            scores[:, token] = scores_before[:, token]
    return scores
