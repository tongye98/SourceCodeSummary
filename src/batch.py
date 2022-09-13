# coding: utf-8
"""
Implemention of a mini-batch
"""
import logging 
import torch 
from torch import Tensor
from src.constants import PAD_ID
from typing import List
from src.helps import make_src_map, align
from src.vocabulary import Vocabulary

logger = logging.getLogger(__name__)

class Batch(object):
    """
    Object for holding a batch of data with mask during training.
    Input is yield from 'collate_fn()' called by torch.data.utils.DataLoader.
    """
    def __init__(self, src, trg, copy_param_list) -> None:
        src_lengths = [len(sentence) for sentence in src] # not include <bos> include <eos>
        max_src_len = max(src_lengths)
        trg_lengths = [len(sentence) for sentence in trg] # include <bos> and <eos>
        max_trg_len = max(trg_lengths)

        padded_src_sentences = []
        for sentence in src:
            pad_number = max_src_len - len(sentence)
            assert pad_number >= 0
            padded_src_sentences.append(sentence + [PAD_ID] * pad_number)
        
        padded_trg_sentences = []
        for sentence in trg:
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
        self.trg_mask = (self.trg_truth != PAD_ID).unsqueeze(1) # [batch_size, 1, trg_length]
        self.ntokens = (self.trg_truth != PAD_ID).data.sum().item()
        
        # for copy mechanism
        self.src_vocabs = list()
        src_maps = list()
        alignments = list()
        # copy_param_list [dict(), dict(), dict()]
        self.copy_param_list = copy_param_list
        for copy_param in self.copy_param_list:
            self.src_vocabs.append(copy_param["src_vocab"])
            src_maps.append(torch.tensor(copy_param["src_map"]))
            alignments.append(torch.tensor(copy_param["alignment"]))

        self.src_maps = make_src_map(src_maps)
        # self.src_maps tensor [batch_size, src_len, extra_words] # no bos, no eos
        self.alignments = align(alignments)
        # self.alignments tensor [batch_size, trg_len] # no bos, but has eos

    def move2cuda(self, device:torch.device):
        """Move batch data to GPU"""
        assert isinstance(device, torch.device)
        assert device.type == "cuda", "device type != cuda"

        self.src = self.src.to(device, non_blocking=True)
        self.src_lengths = self.src_lengths.to(device, non_blocking=True)
        self.src_mask = self.src_mask.to(device, non_blocking=True)

        self.trg_input = self.trg_input.to(device, non_blocking=True)
        self.trg_truth = self.trg_truth.to(device, non_blocking=True)
        self.trg_lengths = self.trg_lengths.to(device, non_blocking=True)
        self.trg_mask = self.trg_mask.to(device, non_blocking=True)

        self.src_maps = self.src_maps.to(device, non_blocking=True)
        self.alignments = self.alignments.to(device, non_blocking=True)

    def normalize(self, tensor, normalization):
        """
        Normalizes batch tensor (i.e. loss)
        """
        if normalization == "sum":
            return tensor 
        elif normalization == "batch":
            normalizer = self.nseqs
        elif normalization == "tokens":
            normalizer = self.ntokens
        elif normalization == "none":
            normalizer = 1
        
        norm_tensor = tensor / normalizer
        return norm_tensor
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(nseqs={self.nseqs}, ntokens={self.ntokens}.)")