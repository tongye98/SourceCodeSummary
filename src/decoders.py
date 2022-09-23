# coding: utf-8
from typing import Tuple
from torch import Tensor, nn 
from src.transformer_layers import PositionalEncoding, TransformerDecoderLayer, LearnablePositionalEncoding
from src.helps import freeze_params, subsequent_mask
import logging  

logger = logging.getLogger(__name__)

class TransformerDecoder(nn.Module):
    """
    Classical Transformer Decoder
    """
    def __init__(self, model_dim:int=512, ff_dim:int=2048,
                 num_layers:int=6, head_count:int=8, dropout:float=0.1,
                 emb_dropout:float=0.1, layer_norm_position:str='pre',
                 trg_pos_emb:str="absolute", max_trg_len:int=512, freeze:bool=False,
                 max_relative_positon:int=0, use_negative_distance:bool=False) -> None:
        super().__init__()

        self.layers = nn.ModuleList([TransformerDecoderLayer(model_dim,ff_dim,head_count, dropout,
            layer_norm_position, max_relative_positon, use_negative_distance) for _ in range(num_layers)])
        
        assert trg_pos_emb in {"absolute", "learnable", "relative"}
        self.trg_pos_emb = trg_pos_emb
        if self.trg_pos_emb == "absolute":
            self.pe = PositionalEncoding(model_dim)
        elif self.trg_pos_emb == "learnable":
            self.lpe = LearnablePositionalEncoding(model_dim, max_trg_len + 2) # add 2 for <bos> <eos>
        else:
            logger.warning("self.trg_pos_emb value need double check")

        self.layer_norm = nn.LayerNorm(model_dim) if layer_norm_position == 'pre' else None
        self.emb_layer_norm = nn.LayerNorm(model_dim) if self.trg_pos_emb == "learnable" else None
        self.emb_dropout = nn.Dropout(emb_dropout)
        
        self.head_count = head_count
        self.layer_norm_position = layer_norm_position
        self.model_dim = model_dim
        if freeze:
            freeze_params(self)
    
    def forward(self, embed_trg:Tensor, encoder_output:Tensor,
                src_mask:Tensor, trg_mask:Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Transformer decoder forward pass.
        embed_trg [batch_size, trg_len, model_dim]
        encoder_ouput [batch_size, src_len, model_dim]
        src_mask [batch_size, 1, src_len]
        trg_mask [batch_size, 1, trg_len]
        return:
            output [batch_size, trg_len, model_dim]  
            cross_attention_weight [batch_size, trg_len, src_len]
        """
        assert trg_mask is not None, "trg mask is required for Transformer decoder"
        if self.trg_pos_emb == "absolute":
            embed_trg = self.pe(embed_trg)
        elif self.trg_pos_emb == "learnable":
            embed_trg = self.lpe(embed_trg)
            # FIXME should layer_norm?
        else:
            embed_trg = embed_trg

        input = self.emb_dropout(embed_trg)

        trg_mask = trg_mask & subsequent_mask(embed_trg.size(1)).type_as(trg_mask)
        # trg_mask [batch_size, 1, trg_len] -> [batch_size, trg_len, trg_len] (include mask the token unseen)

        for layer in self.layers:
            penultimate = input
            input, cross_attention_weight = layer(input, memory=encoder_output, src_mask=src_mask, trg_mask=trg_mask)
        
        penultimate_representation = self.layers[-1].context_representation(penultimate, encoder_output, src_mask, trg_mask)

        if self.layer_norm is not None:
            input = self.layer_norm(input)
        
        output = input
        return output, penultimate_representation, cross_attention_weight
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(num_layers={len(self.layers)}, "
                f"head_count={self.head_count}, " 
                f"layer_norm_position={self.layer_norm_position})")

if __name__ == "__main__":
    # Test decoder
    decoder = TransformerDecoder(512, 2048, 6, 8, 10000, 0.1, 0.1, 'post', False)
    print(decoder)