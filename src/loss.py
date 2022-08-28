# coding: utf-8
"""
Module to implement training loss
"""
import torch 
from torch import Tensor, nn 

class XentLoss(nn.Module):
    """
    Cross-Entropy loss with optional label smoothing
    """
    def __init__(self, pad_index: int, smoothing: float=0.0) -> None:
        super().__init__()
        self.smoothing = smoothing
        self.pad_index = pad_index 
        if self.smoothing <= 0.0:
            self.criterion = nn.NLLLoss(ignore_index=self.pad_index, reduction="sum")
        else:
            self.criterion = nn.KLDivLoss(reduction="sum")
    
    def forward(self, log_probs: Tensor, targets: Tensor) -> Tensor:
        """
        Compute the cross-entropy between logits and targets.
        """
        logits = self.criterion(log_probs, targets)
        return logits
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(criterion={self.criterion}, "
                f"smoothing={self.smoothing})")