# -*- coding: utf-8 -*-
"""
Transformer layers
"""
import math
import torch
import torch.nn as nn
from torch import Tensor
from src.helps import generate_relative_position_matrix
import logging
from src.vocabulary import Vocabulary 

logger = logging.getLogger(__name__)

class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from 'Attention is all you need'.
    consider relative position.
    """
    def __init__(self, head_count: int, model_dim:int, dropout: float=0.1,
                 max_relative_position=0, use_negative_distance=False) -> None:
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
        self.use_negative_distance = use_negative_distance

        if self.max_relative_position > 0:
            relative_position_size = self.max_relative_position*2+1 if self.use_negative_distance is True else self.max_relative_position+1
            self.relative_position_embedding_key = nn.Embedding(relative_position_size, self.head_size)
            self.relative_position_embedding_value = nn.Embedding(relative_position_size, self.head_size)

    def forward(self, key, value, query, mask=None):
        """
        Compute multi-headed attention.
        key  [batch_size, seq_len, hidden_size]
        value[batch_size, seq_len, hidden_size]
        query[batch_size, seq_len, hidden_size]
        mask [batch_size, 1/seq_len, seq_len] (pad position is false)

        return 
            - output [batch_size, query_len, hidden_size]
            - attention_output_weights [batch_size, query_len, key_len]
        """
        batch_size = key.size(0)
        key_len = key.size(1)
        query_len = query.size(1)
        value_len = value.size(1)

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

        if self.max_relative_position > 0: 
            relative_position_matrix = generate_relative_position_matrix(key_len, self.max_relative_position, self.use_negative_distance)
            relative_position_matrix = relative_position_matrix.to(key.device)
            relative_key = self.relative_position_embedding_key(relative_position_matrix)
            # relative_key [key_len, key_len, head_size]
            relative_vaule = self.relative_position_embedding_value(relative_position_matrix)
            # relative_value [value_len, value_len, head_size]
            r_query = query.permute(2,0,1,3).reshape(query_len, batch_size*self.head_count, self.head_size)
            assert query_len == key_len, "For relative position."
            scores_relative = torch.matmul(r_query, relative_key.transpose(1,2)).reshape(query_len, batch_size, self.head_count, key_len)
            scores_relative = scores_relative.permute(1, 2, 0, 3)
            scores = scores + scores_relative

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

        if self.max_relative_position > 0:
            r_attention_probs = attention_probs.permute(2,0,1,3).reshape(query_len, batch_size*self.head_count, key_len)
            context_relative = torch.matmul(r_attention_probs, relative_vaule) # context_relative [query_len, batch_size*self.head_count, head_size]
            context_relative = context_relative.reshape(query_len, batch_size, self.head_count, self.head_size).permute(1, 2, 0, 3)
            context_relative = context_relative.transpose(1, 2).contiguous().view(batch_size, -1, self.head_count*self.head_size)
            # context_relative [batch_size, query_len, hidden_size]
            context = context + context_relative

        output = self.output_layer(context)

        attention_output_weights = attention_weights.view(batch_size, self.head_count, query_len, key_len).sum(dim=1) / self.head_count
        # [batch_size, query_len, key_len]
        return output, attention_output_weights


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-forward layer
    Projects to ff_size and then back dowm to input_dim.
    Pre-LN and Post-LN Transformer cite "Understanding the Difficulity of Training Transformers"
    """
    def __init__(self, model_dim:int, ff_dim:int, dropout:float=0.1, layer_norm_position:str="post") -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(model_dim)
        self.pwff = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, model_dim),
            nn.Dropout(dropout),
        )
        self.layer_norm_position = layer_norm_position
        assert self.layer_norm_position in {"pre","post"}
    
    def forward(self, x:Tensor) -> Tensor:
        """
        Layer definition.
        input x: [batch_size, seq_len, model_dim]
        output: [batch_size, seq_len, model_dim]
        """
        residual = x
        if self.layer_norm_position == "pre":
            x = self.layer_norm(x)
        x = self.pwff(x) + residual
        if self.layer_norm_position == "post":
            x = self.layer_norm(x)
        return x


class PositionalEncoding(nn.Module):
    """
    Pre-compute position encodings.(PE)
    """
    def __init__(self, model_dim:int=0, max_len:int=5000) -> None:
        if model_dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with odd dim!")
        
        pe = torch.zeros(max_len, model_dim) # pe [max_len, model_dim]
        position = torch.arange(0, max_len).unsqueeze(1) # position [max_len,1]
        div_term = torch.exp(torch.arange(0,model_dim,2,dtype=torch.float)*-(math.log(10000.0)/model_dim))
        pe[:,0::2] = torch.sin(position.float() * div_term)
        pe[:,1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0) # pe [1, max_len, model_dim]
        super().__init__()
        self.register_buffer("pe",pe)
    
    def forward(self, emb: Tensor) -> Tensor:
        # emb: [batch_size, seq_len, model_dim]
        return emb + self.pe[:, :emb.size(1)]


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable position encodings. (used in Bert etc.)
    """
    def __init__(self, model_dim:int=0, max_len:int=512) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.max_len = max_len

        self.learn_lut = nn.Embedding(max_len, self.model_dim)

    def forward(self, embed: Tensor) -> Tensor:
        """
        Perform lookup for input(source) in the learnable embeding position table.
        :param src_input [batch_size, src_len]
        :param embed_src [batch_size, src_len, embed_dim]
        return embed_src + lpe(src_input)
        """
        batch_size = embed.size(0)
        len = embed.size(1)
        assert len <= self.max_len, 'len must <= max len'
        position_input = torch.arange(len).unsqueeze(0).repeat(batch_size, 1).to(embed.device)
        # make sure embed and position embed have same scale.
        return embed + self.learn_lut(position_input)

class TransformerEncoderLayer(nn.Module):
    """
    Classical transformer Encoder layer
    containing a Multi-Head attention layer and a position-wise feed-forward layer.
    """
    def __init__(self, model_dim:int, ff_dim:int, head_count:int, 
                 dropout:float=0.1, layer_norm_position:str="pre",
                 max_relative_position:int=0, use_negative_distance:bool=False) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(model_dim)
        self.src_src_attenion = MultiHeadedAttention(head_count, model_dim, dropout, max_relative_position, use_negative_distance)
        self.feed_forward = PositionwiseFeedForward(model_dim, ff_dim, dropout=dropout, layer_norm_position=layer_norm_position)
        self.dropout = nn.Dropout(dropout)

        self.layer_norm_position = layer_norm_position
        assert self.layer_norm_position in {'pre','post'}
    
    def forward(self, input:Tensor, mask:Tensor) -> Tensor:
        """
        Forward pass for a single transformer encoder layer.
        input [batch_size, src_len, model_dim]
        mask [batch_size, 1, src_len]
        return:
            output [batch_size, src_len, model_dim]
        """
        residual = input
        if self.layer_norm_position == "pre":
            input = self.layer_norm(input)
        attention_output, _ = self.src_src_attenion(input, input, input, mask)
        feedforward_input = self.dropout(attention_output) + residual

        if self.layer_norm_position == "post":
            feedforward_input = self.layer_norm(feedforward_input)

        output = self.feed_forward(feedforward_input)
        return output


class TransformerDecoderLayer(nn.Module):
    """
    Classical transformer Decoder Layer
    """
    def __init__(self, model_dim:int, ff_dim:int,head_count:int,
                 dropout:float=0.1, layer_norm_position:str='pre',
                 max_relative_position:int=0, use_negative_distance:bool=False) -> None:
        "layer norm position either 'pre' or 'post'."
        super().__init__()
        self.model_dim = model_dim
        self.trg_trg_attention = MultiHeadedAttention(head_count, model_dim, dropout, max_relative_position, use_negative_distance)
        self.src_trg_attention = MultiHeadedAttention(head_count, model_dim, dropout, max_relative_position=0, use_negative_distance=False)

        self.feed_forward = PositionwiseFeedForward(model_dim, ff_dim, dropout=dropout, layer_norm_position=layer_norm_position)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.layer_norm2 = nn.LayerNorm(model_dim)

        self.layer_norm_position = layer_norm_position
        assert self.layer_norm_position in {'pre','post'}

    def forward(self, input:Tensor, memory:Tensor,
                src_mask: Tensor, trg_mask:Tensor) -> Tensor:
        """
        Forward pass for a single transformer decoer layer.
        input [batch_size, trg_len, model_dim]
        memory [batch_size, src_len, model_dim]
        src_mask [batch_size, 1, src_len]
        trg_mask [batch_size, trg_len, trg_len]
        return:
            output [batch_size, trg_len, model_dim]
            cross_attention_weight [batch_size, trg_len, src_len]
        """
        residual = input
        if self.layer_norm_position == 'pre':
            input = self.layer_norm(input)
        self_attention_output, _ = self.trg_trg_attention(input,input,input, mask=trg_mask)
        cross_attention_input = self.dropout(self_attention_output) + residual

        if self.layer_norm_position == 'post':
            cross_attention_input = self.layer_norm(cross_attention_input)

        cross_residual = cross_attention_input
        if self.layer_norm_position == 'pre':
            cross_attention_input = self.layer_norm2(cross_attention_input)
        cross_attention_output, cross_attention_weight = self.src_trg_attention(memory, memory, cross_attention_input,mask=src_mask)
        feedforward_input = self.dropout(cross_attention_output) + cross_residual

        if self.layer_norm_position == 'post':
            feedforward_input = self.layer_norm2(feedforward_input)

        output = self.feed_forward(feedforward_input)
        return output, cross_attention_weight
    
    def context_representation(self, penultimate:Tensor, encoder_output:Tensor, src_mask:Tensor, trg_mask:Tensor) -> Tensor:
        """
        Get the hidden state for search purpose.
        The hidden state means the token semantic.
        """
        residual = penultimate
        if self.layer_norm_position == 'pre':
            penultimate = self.layer_norm(penultimate)
        self_attention_output, _ = self.trg_trg_attention(penultimate, penultimate, penultimate, trg_mask)
        cross_attention_input = self.dropout(self_attention_output) + residual

        if self.layer_norm_position == 'post':
            cross_attention_input = self.layer_norm(cross_attention_input)
        
        cross_residual = cross_attention_input
        if self.layer_norm_position == 'pre':
            cross_attention_input = self.layer_norm2(cross_attention_input)
        cross_attention_output, cross_attention_weight = self.src_trg_attention(encoder_output, encoder_output, cross_attention_input, src_mask)
        feedforward_input = self.dropout(cross_attention_output) + cross_residual

        if self.layer_norm_position == "post":
            feedforward_input = self.layer_norm2(feedforward_input)
            
        representation = self.feed_forward.layer_norm(feedforward_input)
        return representation
        