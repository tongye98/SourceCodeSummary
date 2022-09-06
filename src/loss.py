# coding: utf-8
"""
Module to implement training loss
"""
import torch 
from torch import Tensor, nn 

class XentLoss(nn.Module):
    """
    Cross-Entropy loss with optional label smoothing
    reduction='sum' means add all sequences and all tokens loss in the batch.
    reduction='mean' means take average of all sequence and all token loss in the batch.
    """
    def __init__(self, pad_index: int, smoothing: float=0.0) -> None:
        super().__init__()
        self.smoothing = smoothing
        self.pad_index = pad_index 
        if self.smoothing <= 0.0:
            self.criterion = nn.NLLLoss(ignore_index=self.pad_index, reduction="sum")
        else:
            self.criterion = nn.KLDivLoss(reduction="sum")
    
    def reshape(self, log_probs: Tensor, target: Tensor) -> Tensor:
        """
        Reshape Tensor because of the input restrict of nn.NLLLoss/nn.CrossEntropyLoss
        :param log_probs [batch_size, trg_len, vocab_size]
        :param target [batch_size, trg_len]
        """
        vocab_size = log_probs.size(-1)
        log_probs = log_probs.contiguous().view(-1, vocab_size)
        # log_probs [batch_size*trg_len, vocab_size]

        target = target.contiguous().view(-1)
        # target [batch_size*trg_len]

        return log_probs, target
    
    def forward(self, log_probs: Tensor, target: Tensor) -> Tensor:
        """
        Compute the cross-entropy between logits and targets.
        :param log_probs [batch_size, trg_len, vocab_size]
        :param target [batch_size, trg_len]
        """
        log_probs, target = self.reshape(log_probs, target)
        logits = self.criterion(log_probs, target)
        return logits
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(criterion={self.criterion}, "
                f"smoothing={self.smoothing})")


class CopyGeneratorCriterion(nn.Module):
    pass