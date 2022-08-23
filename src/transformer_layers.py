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

        self.key_project = nn.Linear(model_dim, head_count * self.head_size)
        self.query_project = nn.Linear(model_dim, head_count * self.head_size)
        self.value_project = nn.Linear(model_dim, head_count * self.head_size)

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.output_layer = nn.Linear(model_dim, model_dim)

        self.max_relative_position = max_relative_position
        self.use_neg_dist = use_neg_dist
        self._coverage = coverage
    

    def forward(self, key, value, query, mask=None):
        """
        Compute multi-headed attention.
        key  [batch_size, seq_len, hidden_size]
        value[batch_size, seq_len, hidden_size]
        query[batch_size, seq_len, hidden_size]
        mask [batch_size, 1, seq_len] (pad position is false)

        return 
            - output [batch_size, query_len, hidden_size]
            - attention_output_weights [batch_size, query_len, key_len]
        """
        batch_size = key.size(0)
        key_len = key.size(1)
        query_len = query.size(1)

        # project query key value
        key = self.key_project(key)
        value = self.value_project(value)
        query = self.query_project(query)

        #reshape key, value, query 
        key = key.view(batch_size, -1, self.head_count, self.head_size).transpose(1,2) #[batch_size, head_count, key_len, head_size]
        value = value.view(batch_size, -1, self.head_count, self.head_size).transpose(1,2)
        query = query.view(batch_size, -1, self.head_count, self.head_size).transpose(1,2)

        # scale and calculate attention scores
        query = query / math.sqrt(self.head_size)
        scores = torch.matmul(query, key.transpose(2,3))
        # scores [batch_size, head_count, query_len, key_len]

        # apply mask Note: add a dimension to mask, -> [batch_size, 1, 1, key_len]
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        
        # apply attention dropout
        attention_weights = self.softmax(scores) # attention_weights [batch_size, head_count, query_len, key_len]
        attention_probs = self.dropout(attention_weights)

        # get context vector
        context = torch.matmul(attention_probs, value) # context [batch_size, head_count, query_len, head_size]
        context = context.transpose(1,2).contiguous().view(batch_size, -1, self.head_count*self.head_size)
        # context [batch_size, query_len, hidden_size]

        output = self.output_layer(context)

        attention_output_weights = attention_weights.view(batch_size, self.head_count, query_len, key_len).sum(dim=1) / self.head_count
        # [batch_size, query_len, key_len]
        return output, attention_output_weights




        
