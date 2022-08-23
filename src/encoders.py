from torch import Tensor, nn 
from transformer_layers import TransformerEncoderLayer, PositionalEncoding
from helps import freeze_params

class TransformerEncoder(nn.Module):
    """
    Classical Transformer Encoder.
    """
    def __init__(self, model_dim:int=512, ff_dim:int=2048, 
                 num_layers:int=6, head_count:int=8, dropout:float=0.1, 
                 layer_norm_position:str='post', emb_dropout:float=0.1, freeze:bool=False) -> None:
        super().__init__()

        self.layers = nn.ModuleList([TransformerEncoderLayer(model_dim,ff_dim,head_count,dropout=dropout,
                                        layer_norm_position=layer_norm_position) for _ in range(num_layers)])
        
        # FIXME necessary to add positonal encoding or relative encoding
        self.pe = PositionalEncoding(model_dim)
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.layer_norm = nn.LayerNorm(model_dim,eps=1e-6) if layer_norm_position == 'pre' else None
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
        # FIXME  add positon embedding
        # TODO get each layer representation and attention score
        input = self.emb_dropout(embed_src)
        for layer in self.layers:
            input = layer(input, mask)
        
        if self.layer_norm is not None: # for Pre-LN Transformer
            input = self.layer_norm(input) 
        
        return input
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(num_layers={len(self.layers)}, "
                f"head_count={self.head_count}, " 
                f"layer_norm_position={self.layer_norm_position}")