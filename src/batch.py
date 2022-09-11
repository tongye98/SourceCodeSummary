# coding: utf-8
"""
Implemention of a mini-batch
"""

import logging 
import torch 
from torch import Tensor
from src.constants import PAD_ID
from typing import List
from src.vocabulary import Vocabulary

logger = logging.getLogger(__name__)


class Batch(object):
    """
    Object for holding a batch of data with mask during training.
    Input is yield from 'collate_fn()' called by torch.data.utils.DataLoader.
    """
    def __init__(self, src, trg, device, src_vocabs:List[Vocabulary]=None, source_maps:List[Tensor]=None, alignments:List[Tensor]=None) -> None:
        self.device = device

        max_src_len = max([len(sentence) for sentence in src])
        max_trg_len = max([len(sentence) for sentence in trg])

        padded_src_sentences = []
        src_lengths = []
        for sentence in src:
            src_lengths.append(len(sentence))
            pad_number = max_src_len - len(sentence)
            assert pad_number >= 0
            padded_src_sentences.append(sentence + [PAD_ID] * pad_number)
        
        padded_trg_sentences = []
        trg_lengths = []
        for sentence in trg:
            trg_lengths.append(len(sentence))
            pad_number = max_trg_len -len(sentence)
            assert pad_number >= 0
            padded_trg_sentences.append(sentence + [PAD_ID] * pad_number)
        
        self.src = torch.tensor(padded_src_sentences).long()
        self.src_lengths = torch.tensor(src_lengths).long()
        self.src_mask = (self.src != PAD_ID).unsqueeze(1)
        # src_mask unpad is true, pad is false (batch, 1, pad_src_length)
        self.nseqs = self.src.size(0)
        
        self.trg = torch.tensor(padded_trg_sentences).long()
        self.trg_input = self.trg[:, :-1]
        self.trg_truth = self.trg[:, 1:] # self.trg_truth is used for loss computation
        self.trg_lengths = torch.tensor(trg_lengths).long() - 1 # original trg length + 1
        self.trg_mask = (self.trg_truth != PAD_ID).unsqueeze(1) #[batch_size, 1, trg_length]
        self.ntokens = (self.trg_truth != PAD_ID).data.sum().item()


        # # example: trg -> <bos> a b <pad> <eos>
        # self.trg_input = trg[:, :-1] # [batch_size, modified_trg_length-1]  modified_trg_length = original_trg_len + 2
        # self.trg = trg[:, 1:] 
        # # trg(= trg_truth) is used for loss computation, shifted by 1 since BOS token.
        # self.trg_length = trg_length - 1 # trg length for train
        
        # self.trg_mask = (self.trg != PAD_ID).unsqueeze(1) # [batch_size, 1, trg_length]
        # self.ntokens = (self.trg != PAD_ID).data.sum().item()
        
        # # for copy mechanism
        # self.src_vocabs = src_vocabs
        # self.source_maps = source_maps
        # # source_maps [batch_size, src_len, extra_words]
        # self.alignments = alignments
        # # alignments [batch_size, original_target_length+1]  no bos, but has eos

        # if device.type == "cuda":
        #     self.move2cuda(device)
    
    def move2cuda(self):
        """Move batch to GPU"""
        device = self.device
        assert isinstance(device, torch.device)
        assert self.device.type == "cuda", "device type != cuda"

        self.src = self.src.to(device)
        self.src_lengths = self.src_lengths.to(device)
        self.src_mask = self.src_mask.to(device)

        self.trg_input = self.trg_input.to(device)
        self.trg_truth = self.trg_truth.to(device)
        self.trg_mask = self.trg_mask.to(device)

        # self.source_maps = self.source_maps.to(device)
        # self.alignments = self.alignments.to(device)

    
    def normalize(self, tensor, normalization, n_gpu):
        """
        Normalizes batch tensor (i.e. loss)
        """
        if n_gpu > 1:
            tensor = tensor.sum()
        
        if normalization == "sum":
            return tensor 
        elif normalization == "batch":
            normalizer = self.nseqs
        elif normalization == "tokens":
            normalizer = self.ntokens
        elif normalization == "none":
            normalizer = 1
        
        norm_tensor = tensor / normalizer

        # FIXME Already normalization ?
        if n_gpu > 1:
            norm_tensor = norm_tensor / n_gpu

        return norm_tensor
    
    def sort_by_length(self):
        """
        Sort by src length(descending) and return index to revert sort
        """
        # FIXME IS this function necessary? 
        # for rnn, lstm is necessary, but for transformer is not necessary.
        pass

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(nseqs={self.nseqs}, ntokens={self.ntokens}.)")