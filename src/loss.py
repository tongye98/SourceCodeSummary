# coding: utf-8
"""
Module to implement training loss
"""
import torch 
from torch import Tensor, nn
from torch.autograd import Variable
import logging

logger = logging.getLogger(__name__)

class XentLoss(nn.Module):
    """
    Cross-Entropy loss with optional label smoothing.
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

        if self.smoothing > 0:
            target = self.smooth_target(target.contiguous().view(-1), vocab_size)
        else:
            target = target.contiguous().view(-1)
            # target [batch_size*trg_len]

        return log_probs, target

    def smooth_target(self, target, vocab_size):
        """
        target: [batch_size*trg_len]
        vocab_size: a number
        return: smoothed target distributions, batch*trg_len x vocab_size
        """
        # batch*trg_len x vocab_size
        smooth_dist = target.new_zeros((target.size(0), vocab_size)).float()
        # fill distribution uniformly with smoothing
        smooth_dist.fill_(self.smoothing / (vocab_size - 2))
        # assign true label the probability of 1-smoothing ("confidence")
        smooth_dist.scatter_(1, target.unsqueeze(1).data, 1.0 - self.smoothing)
        # give padding probability of 0 everywhere
        smooth_dist[:, self.pad_index] = 0
        # masking out padding area (sum of probabilities for padding area = 0)
        padding_positions = torch.nonzero(target.data == self.pad_index,
                                          as_tuple=False)
        if len(padding_positions) > 0:
            smooth_dist.index_fill_(0, padding_positions.squeeze(), 0.0)
        return Variable(smooth_dist, requires_grad=False)

    def forward(self, log_probs: Tensor, target: Tensor) -> Tensor:
        """
        Compute the cross-entropy between logits and targets.
        :param log_probs [batch_size, trg_len, vocab_size]
        :param target [batch_size, trg_len]
        """
        log_probs, target = self.reshape(log_probs, target)
        batch_loss = self.criterion(log_probs, target)
        return batch_loss
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(criterion={self.criterion}, "
                f"smoothing={self.smoothing})")