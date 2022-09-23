from torch import Tensor, nn 
from src.transformer_layers import TransformerEncoderLayer, PositionalEncoding, LearnablePositionalEncoding
from src.helps import freeze_params
import logging 

logger = logging.getLogger(__name__)

class TransformerEncoder(nn.Module):
    """
    Classical Transformer Encoder.
    """
    def __init__(self, model_dim:int=512, ff_dim:int=2048, 
                 num_layers:int=6, head_count:int=8, dropout:float=0.1, 
                 emb_dropout:float=0.1, layer_norm_position:str='pre', 
                 src_pos_emb:str="absolute", max_src_len:int=512, freeze:bool=False,
                 max_relative_position:int=0, use_negative_distance:bool=False) -> None:
        super().__init__()

        self.layers = nn.ModuleList([TransformerEncoderLayer(model_dim, ff_dim, head_count, dropout,
                    layer_norm_position, max_relative_position, use_negative_distance) for _ in range(num_layers)])
        
        assert src_pos_emb in {"absolute", "learnable", "relative"}
        self.src_pos_emb = src_pos_emb
        if self.src_pos_emb == "absolute":
            self.pe = PositionalEncoding(model_dim)
        elif self.src_pos_emb == "learnable":
            self.lpe = LearnablePositionalEncoding(model_dim, max_src_len + 2) # add 2 for <bos> <eos>
        else:
            logger.warning("self.src_pos_emb value need double check")

        self.emb_dropout = nn.Dropout(emb_dropout)
        self.emb_layer_norm = nn.LayerNorm(model_dim) if self.src_pos_emb == "learnable" else None
        self.layer_norm = nn.LayerNorm(model_dim) if layer_norm_position == 'pre' else None
        self.head_count = head_count
        self.layer_norm_position = layer_norm_position
        if freeze:
            freeze_params(self)
    
    def forward(self, embed_src:Tensor, mask:Tensor=None) -> Tensor:
        """
        Pass the input and mask through each layer in turn.
        embed_src [batch_size, src_len, embed_size]
        mask: indicates padding areas (zeros where padding) [batch_size, 1, src_len]
        """
        if self.src_pos_emb == "absolute":
            embed_src = self.pe(embed_src)  # add absolute position encoding
        elif self.src_pos_emb == "learnable":
            embed_src = self.lpe(embed_src)
            # FIXME should layer_norm?
            if self.emb_layer_norm is not None:
                embed_src = self.emb_layer_norm(embed_src)
        else:
            embed_src = embed_src
            
        input = self.emb_dropout(embed_src)
        for layer in self.layers:
            input = layer(input, mask)
        
        if self.layer_norm is not None: # for Pre-LN Transformer
            input = self.layer_norm(input) 
        
        return input
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(num_layers={len(self.layers)}, "
                f"head_count={self.head_count}, " 
                f"layer_norm_position={self.layer_norm_position})")


if __name__ == "__main__":
    # test encoder
    encoder = TransformerEncoder(512,2048,6,8,0.1,0.1,'post',False)
    print(encoder)