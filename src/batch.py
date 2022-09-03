# coding: utf-8
"""
Implemention of a mini-batch
"""

import logging 
import torch 
from torch import Tensor
from src.constants import PAD_ID

logger = logging.getLogger(__name__)


class Batch(object):
    """
    Object for holding a batch of data with mask during training.
    Input is yield from 'collate_fn()' called by torch.data.utils.DataLoader.
    """
    def __init__(self, src:Tensor, src_length: Tensor,
                 trg: Tensor, trg_length: Tensor, device: torch.device) -> None:
        
        self.src = src
        self.src_length = src_length
        self.src_mask = (self.src != PAD_ID).unsqueeze(1)
        # src_mask unpad is true, pad is false (batch, 1, pad_src_length)
        self.nseqs = self.src.size(0)

        # example: trg -> <bos> a b <pad> <eos>
        self.trg_input = trg[:, :-1] # [batch_size, trg_length-1]
        self.trg = trg[:, 1:] 
        # trg(trg_truth) is used for loss computation, shifted by 1 since BOS token.
        self.trg_length = trg_length - 1 # FIXME why is 1? 

        self.trg_mask = (self.trg != PAD_ID).unsqueeze(1) # [batch_size, 1, trg_length]
        self.ntokens = (self.trg != PAD_ID).data.sum().item()

        if device.type == "cuda":
            self.move2cuda(device)
    
    def move2cuda(self, device: torch.device):
        """Move batch to GPU"""
        assert isinstance(device, torch.device)
        self.src = self.src.to(device)
        self.src_length = self.src_length.to(device)
        self.src_mask = self.src_mask.to(device)

        self.trg_input = self.trg_input.to(device)
        self.trg = self.trg.to(device)
        self.trg_mask = self.trg_mask.to(device)

    
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