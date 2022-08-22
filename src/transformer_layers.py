# -*- coding: utf-8 -*-
"""
Transformer layers
"""
import math
import torch
import torch.nn as nn
from torch import Tensor

class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from 'Attention is all you need'.
    consider relative position.
    """
    def __init__(self, head_count: int, model_dim:int, dropout: float=0.1,
                 max_relative_position=0, use_neg_dist=True, coverage=False) -> None:
        super().__init__()
        assert model_dim % head_count == 0, 'model dim must be divisible by head count'

        self.head_size = model_dim // head_count
        self.head_count = head_count
        self.model_dim = model_dim

        self.key = nn.Linear(model_dim, head_count * self.head_size)
        self.query = nn.Linear(model_dim, head_count * self.head_size)
        self.value = nn.Linear(model_dim, head_count * self.head_size)

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.output = nn.Linear(model_dim, model_dim)

        self.max_relative_position = max_relative_position
        self.use_neg_dist = use_neg_dist
        self._coverage = coverage
    

    def forward(self, key, value, query, mask=None):
        """
        Compute multi-headed attention.
        
        """
        pass
