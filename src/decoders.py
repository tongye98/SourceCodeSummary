# coding: utf-8
from torch import Tensor, nn 
from src.transformer_layers import PositionalEncoding, TransformerDecoderLayer 
from src.helps import freeze_params, subsequent_mask
import logging  

logger = logging.getLogger(__name__)

class TransformerDecoder(nn.Module):
    """
    Classical Transformer Decoder
    """
    def __init__(self, model_dim:int=512, ff_dim:int=2048,
                 num_layers:int=6, head_count:int=8, vocab_size:int=1, dropout:float=0.1,
                 emb_dropout:float=0.1, layer_norm_position:str='post',freeze:bool=False) -> None:
        super().__init__()

        self.layers = nn.ModuleList([TransformerDecoderLayer(model_dim,ff_dim,head_count,dropout=dropout,
                                                            layer_norm_position=layer_norm_position) for _ in range(num_layers)])
        self.pe = PositionalEncoding(model_dim)
        self.layer_norm = nn.LayerNorm(model_dim,eps=1e-6) if layer_norm_position == 'pre' else None
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.output_layer = nn.Linear(model_dim, vocab_size, bias=False)
        
        self.head_count = head_count
        self.layer_norm_position = layer_norm_position
        self.output_size = vocab_size
        if freeze:
            freeze_params(self)
    
    def forward(self, embed_trg:Tensor,encoder_output:Tensor,
                src_mask:Tensor,trg_mask:Tensor):
        """
        Transformer decoder forward pass.
        embed_trg [batch_size, trg_len, model_dim]
        encoder_ouput [batch_size, src_len, model_dim]
        src_mask [batch_size, 1, src_len]
        trg_mask [batch_size, 1, trg_len]
        return:
            output [batch_size, trg_len, vocab_size]    after output layer
            input [batch_size, trg_len, model_dim]      before output layer
            cross_attention_weight [batch_size, trg_len, src_len]
        """
        assert trg_mask is not None, "trg mask is required for Transformer decoder"
        embed_trg = self.pe(embed_trg)
        input = self.emb_dropout(embed_trg)

        trg_mask = trg_mask & subsequent_mask(embed_trg.size(1)).type_as(trg_mask)
        # trg_mask [batch_size, 1, trg_len] -> [batch_size, trg_len, trg_len] (include mask the token unseen)

        for layer in self.layers:
            input, cross_attention_weight = layer(input, memory=encoder_output, src_mask=src_mask, trg_mask=trg_mask)
        
        if self.layer_norm is not None:
            input = self.layer_norm(input)
        
        output = self.output_layer(input)
        return output, input, cross_attention_weight
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(num_layers={len(self.layers)}, "
                f"head_count={self.head_count}, " 
                f"layer_norm_position={self.layer_norm_position})")

if __name__ == "__main__":
    # Test decoder
    decoder = TransformerDecoder(512, 2048, 6, 8, 10000, 0.1, 0.1, 'post', False)
    print(decoder)