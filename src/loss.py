# coding: utf-8
"""
Module to implement training loss
"""
import torch 
from torch import Tensor, nn
from src.constants import UNK_ID, PAD_ID
import torch.nn.functional as F
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
        batch_loss = self.criterion(log_probs, target)
        return batch_loss
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(criterion={self.criterion}, "
                f"smoothing={self.smoothing})")


class CopyGeneratorLoss(nn.Module):
    def __init__(self, trg_vocab_size, force_copy) -> None:
        super().__init__()
        self.force_copy = force_copy
        self.offset = trg_vocab_size
        self.eps = 1e-20
    
    def reshape(self, prob:Tensor, alignment:Tensor, target:Tensor):
        """
        :param prob [batch_size, trg_len, trg_vocab_size + extra_words]
        :param alignment [batch_size, trg_len]
        :param target [batch_size, trg_len]
        """
        total_vocab_size = prob.size(-1)
        prob = prob.contiguous().view(-1, total_vocab_size)
        # prob [batch_size*trg_len, trg_vocab_size+extra_words]
        alignment = alignment.view(-1)
        # alignment [batch_size*trg_len]
        target = target.view(-1)
        # target [batch_size*trg_len]
        return prob, alignment, target

    def forward(self, prob:Tensor, alignment:Tensor, target:Tensor):
        """
        :param prob [batch_size, trg_len, trg_vocab_size + extra_words]
        :param alignment [batch_size, trg_len]
        :param target [batch_size, trg_len]
        """
        batch_size = target.size(0)
        trg_len = target.size(1)
        target_not_pad = target.ne(PAD_ID)
        assert trg_len == alignment.size(1)

        # reshape
        prob, alignment, target = self.reshape(prob, alignment, target)
        # prob [batch_size*trg_len, trg_vocab_size+extra_words]
        # alignment [batch_size*trg_len]
        # target [batch_size*trg_len]

        alignment_unk = alignment.eq(UNK_ID)
        alignment_not_unk = alignment.ne(UNK_ID)
        target_unk = target.eq(UNK_ID)
        target_not_unk = target.ne(UNK_ID)

        extra_select_probs = torch.gather(prob, 1, alignment.view(-1,1) + self.offset).view(-1)
        # extra_select_probs [batch_size*trg_len]
        extra_select_probs = torch.mul(extra_select_probs, alignment_not_unk) + self.eps
        origin_select_probs = torch.gather(prob, 1, target.view(-1,1)).view(-1)
        # in_select_probs [batch_size*trg_len]

        if not self.force_copy:
            # add prob for non-unks in target
            final_prob = extra_select_probs + torch.mul(origin_select_probs, target_not_unk)
            # add prob for when word is unk in both align(src) and traget
            # model need to generate unk token
            final_prob = final_prob + origin_select_probs.mul(alignment_unk).mul(target_unk)
        else:
            # only prob for non-copied tokens
            # In trg, for those token in src, ignore the probs in trg
            # means only consider extra_select_probs
            final_prob = extra_select_probs + torch.mul(origin_select_probs, alignment_unk)
        # final prob [batch_size*trg_len]

        loss = (-final_prob.log()).view(batch_size, trg_len)
        # loss [batch_size, trg_len]
        loss = torch.mul(loss, target_not_pad)
        # batch loss is the sum of all batch sentences and all tokens in the sentence.
        batch_loss = torch.sum(loss)
        return batch_loss