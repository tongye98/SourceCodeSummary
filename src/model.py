import logging
import math
import torch 
from torch import Tensor, nn
import torch.nn.functional as F
from typing import Dict, Tuple
from pathlib import Path
import numpy as np
from src.transformer_layers import TransformerEncoderLayer, PositionalEncoding
from src.transformer_layers import TransformerDecoderLayer, LearnablePositionalEncoding
from src.transformer_layers import GNNEncoderLayer
from src.vocabulary import Vocabulary
from src.helps import ConfigurationError, freeze_params, subsequent_mask
from src.loss import XentLoss

logger = logging.getLogger(__name__)

class Embeddings(nn.Module):
    def __init__(self, embedding_dim:int=64,
                 scale: bool=True, vocab_size:int=0,
                 padding_index:int=1, freeze:bool=False) -> None:
        """
        scale: for transformer see "https://zhuanlan.zhihu.com/p/442509602"
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.scale = scale
        self.vocab_size = vocab_size
        self.lut = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=padding_index)
        if freeze:
            freeze_params(self)

    def forward(self, source: Tensor) -> Tensor:
        """
        Perform lookup for input(source) in the embedding table.
        return the embedded representation for source.
        """
        if self.scale:
            return self.lut(source) * math.sqrt(self.embedding_dim)
        else:
            return self.lut(source)

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"embedding_dim={self.embedding_dim}, "
                f"vocab_size={self.vocab_size})")
    
    def load_from_file(self, embed_path:Path, vocab: Vocabulary) -> None:
        """
        Load pretrained embedding weights from text file.
        - First line is expeceted to contain vocabulary size and dimension.
        The dimension has to match the model's specified embedding size, the vocabulary size is used in logging only.
        - Each line should contain word and embedding weights separated by spaces.
        - The pretrained vocabulary items that are not part of the vocabulary will be ignored (not loaded from the file).
        - The initialization of Vocabulary items that are not part of the pretrained vocabulary will be kept
        - This function should be called after initialization!
        Examples:
            2 5
            the -0.0230 -0.0264  0.0287  0.0171  0.1403
            at -0.0395 -0.1286  0.0275  0.0254 -0.0932        
        """
        embed_dict: Dict[int,Tensor] = {}
        with embed_path.open("r", encoding="utf-8",errors="ignore") as f_embed:
            vocab_size, dimension = map(int, f_embed.readline().split())
            assert self.embedding_dim == dimension, "Embedding dimension doesn't match."
            for line in f_embed.readlines():
                tokens = line.rstrip().split(" ")
                if tokens[0] in vocab.specials or not vocab.is_unk(tokens[0]):
                    embed_dict[vocab.lookup(tokens[0])] = torch.FloatTensor([float(t) for t in tokens[1:]])
            
            logging.info("Loaded %d of %d (%%) tokens in the pre-trained file.",
            len(embed_dict), vocab_size, len(embed_dict)/vocab_size)

            for idx, weights in embed_dict.items():
                if idx < self.vocab_size:
                    assert self.embedding_dim == len(weights)
                    self.lut.weight.data[idx] == weights
            
            logging.info("Cover %d of %d (%%) tokens in the Original Vocabulary.",
            len(embed_dict), len(vocab), len(embed_dict)/len(vocab))

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
            self.lpe = LearnablePositionalEncoding(model_dim, max_src_len + 1) # add 1 for <eos>
        else:
            logger.info("src_pos_emb = {}".format(src_pos_emb))

        self.emb_dropout = nn.Dropout(emb_dropout)
        # self.emb_layer_norm = nn.LayerNorm(model_dim) if self.src_pos_emb == "learnable" else None
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

class GNNEncoder(nn.Module):
    def __init__(self, gnn_type, aggr, model_dim, num_layers, emb_dropout=0.2, residual=False) -> None:
        super().__init__()
        self.layers = nn.ModuleList([GNNEncoderLayer(model_dim=model_dim, GNN=gnn_type, aggr=aggr, residual=residual)
                                      for _ in range(num_layers)])
        
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.layernorm = nn.LayerNorm(model_dim)
    
    def forward(self, node_feature, edge_index, node_batch):
        """
        Input: 
            node_feature: [node_number, node_dim]
            edge_index: [2, edge_number]
            node_batch: {0,0, 1, ..., B-1} | indicate node in which graph. 
                        B: the batch_size or graphs.
        Return 
            output: [batch, Nmax, node_dim]
            mask: [batch, Nmax] bool
            Nmax: max node number in a batch.
        """
        node_feature = self.emb_dropout(node_feature)

        for layer in self.layers:
            node_feature = layer(node_feature, edge_index)
        
        node_feature = self.layernorm(node_feature)

        if node_batch is not None:
            output, mask = to_dense_batch(node_feature, batch=node_batch)

        return output, mask


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
        else:
            embed_trg = embed_trg

        input = self.emb_dropout(embed_trg)

        trg_mask = trg_mask & subsequent_mask(embed_trg.size(1)).type_as(trg_mask)
        # trg_mask [batch_size, 1, trg_len] -> [batch_size, trg_len, trg_len] (include mask the token unseen)

        penultimate = None
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

class Transformer(nn.Module):
    """
    Transformer Model
    """
    def __init__(self, encoder: TransformerEncoder, decoder:TransformerDecoder,
                 src_embed: Embeddings, trg_embed: Embeddings,
                 src_vocab: Vocabulary, trg_vocab: Vocabulary) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.pad_index = self.trg_vocab.pad_index
        self.unk_index = self.trg_vocab.unk_index
        self.bos_index = self.trg_vocab.bos_index
        self.eos_index = self.trg_vocab.eos_index
        self.trg_vocab_size = len(trg_vocab)
        self.model_dim = self.decoder.model_dim
        # self.reduce_dimension = nn.Linear(768, self.model_dim, bias=False)
        self.output_layer = nn.Linear(self.model_dim, self.trg_vocab_size, bias=False)

        self.loss_function = XentLoss(pad_index=self.pad_index, smoothing=0)

    def forward(self, return_type: str=None,
                src_input:Tensor=None, trg_input:Tensor=None,
                src_mask:Tensor=None, trg_mask:Tensor=None,
                encoder_output: Tensor=None, trg_truth:Tensor=None):
        """
        return_type: one of {"loss", "encode", "decode"}
        src_input [batch_size, src_len]
        trg_input [batch_size, trg_len]
        src_mask [batch_size, 1, src_len]
        trg_mask [batch_size, trg_len, trg_len]
        encoder_output [batch_size, src_len, model_dim]
        """
        if return_type == "loss":
            assert trg_input is not None and trg_mask is not None
            encode_output = self.encode(src_input, src_mask)
            decode_output, _, cross_attention_weight = self.decode(trg_input, encode_output, src_mask, trg_mask)
            # decode_output [batch_size, trg_len, model_dim]
            logits = self.output_layer(decode_output)
            # logits [batch_size, trg_len, trg_vocab_size]
            log_probs = F.log_softmax(logits, dim=-1)
            # NOTE batch loss = sum over all sentences of all tokens in the batch that are not pad
            batch_loss = self.loss_function(log_probs, target=trg_truth)
            # logger.warning("batch_loss = {}".format(batch_loss))
            # assert False
            return batch_loss

        elif return_type == "encode":
            assert src_input is not None and src_mask is not None
            return self.encode(src_input, src_mask)
        

        elif return_type == "decode":
            assert trg_input is not None and trg_mask is not None
            return self.decode(trg_input, encoder_output, src_mask, trg_mask)

        elif return_type == "encode_decode":
            assert src_input is not None and src_mask is not None
            assert trg_input is not None and trg_mask is not None 
            encode_output = self.encode(src_input, src_mask)
            decode_output, penultimate_representation, cross_attention_weight = self.decode(trg_input, encode_output, src_mask, trg_mask)
            return decode_output, penultimate_representation, cross_attention_weight, encode_output

        else:
            raise ValueError("return_type must be one of {'loss', 'encode', 'decode', 'encode_decode', 'retrieval_loss'}")
    
    def codebert_encode(self, src_input: Tensor, src_mask: Tensor):
        input = dict()
        input["input_ids"] = src_input
        input["attention_mask"] = src_mask
        codebert_output = self.encoder(**input)
        # logger.warning("encode output shape = {}".format(output))
        # logger.warning("last_hidden_state = {}".format(output.last_hidden_state.shape))
        # logger.warning("pooler output = {}".format(output.pooler_output.shape)) # [batch_size, 768]
        output = codebert_output.last_hidden_state 
        # [batch_size, src_length, 768]
        return self.reduce_dimension(output)

    def encode(self, src_input: Tensor, src_mask:Tensor):
        """
        src_input: [batch_size, src_len]
        src_mask: [batch_size,1,src_len]
        return:
            output [batch_size, src_len, model_dim]
        """
        embed_src = self.src_embed(src_input)  # embed_src [batch_size, src_len, embed_size]
        output = self.encoder(embed_src, src_mask)
        return output
    
    def decode(self, trg_input:Tensor, encode_output:Tensor,
               src_mask:Tensor, trg_mask:Tensor):
        """
        trg_input: [batch_size, trg_len]
        encoder_output: [batch_size, src_len, model_dim]
        src_mask: [batch_size, 1, src_len]
        trg_mask: [batch_size, trg_len, trg_len]
        return:
            output [batch_size, trg_len, model_dim]
            cross_attention_weight [batch_size, trg_len, src_len]
        """
        embed_trg = self.trg_embed(trg_input) # embed_trg [batch_size, trg_len, embed_dim]
        output, penultimate_representation, cross_attention_weight = self.decoder(embed_trg, encode_output, src_mask, trg_mask)
        return output, penultimate_representation, cross_attention_weight

    def __repr__(self) -> str:
        """
        String representation: a description of Encoder, Decoder, Embeddings.
        """
        return (f"{self.__class__.__name__}(\n"
                f"\tencoder={self.encoder},\n"
                f"\tdecoder={self.decoder},\n"
                f"\tsrc_embed={self.src_embed},\n"
                f"\ttrg_embed={self.trg_embed},\n"
                f"\tloss_function={self.loss_function})")
    
    def log_parameters_list(self) -> None:
        """
        Write all model parameters (name, shape) to the log.
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        logger.info("Total parameters number: %d", n_params)
        trainable_parameters = [(name, param) for (name, param) in self.named_parameters() if param.requires_grad]
        for item in trainable_parameters:
            logger.debug("Trainable parameters(name): {0:<60} {1}".format(item[0], str(list(item[1].shape))))
        assert trainable_parameters

def build_model(model_cfg: dict=None, src_vocab: Vocabulary=None,
                trg_vocab: Vocabulary=None, codebert_encode=None) -> Transformer:
    """
    Build and initialize the transformer according to the configuration.
    cfg: yaml model part.
    """
    logger.info("Building Transformer Model...")
    encoder_cfg = model_cfg["encoder"]
    decoder_cfg = model_cfg["decoder"]

    src_pad_index = src_vocab.pad_index
    # src_pad_index = src_vocab.pad_token_id
    trg_pad_index = trg_vocab.pad_index
    assert src_pad_index == trg_pad_index

    src_embed = Embeddings(embedding_dim=encoder_cfg['embeddings']['embedding_dim'],
                           scale=encoder_cfg['embeddings']['scale'],
                           vocab_size=len(src_vocab),
                           padding_index=src_pad_index,
                           freeze=encoder_cfg['embeddings']['freeze'])

    trg_embed = Embeddings(embedding_dim=decoder_cfg['embeddings']['embedding_dim'],
                           scale=decoder_cfg['embeddings']['scale'],
                           vocab_size=len(trg_vocab),
                           padding_index=trg_pad_index,
                           freeze=decoder_cfg['embeddings']['freeze'])    
    
    # Build encoder
    encoder_dropout = encoder_cfg.get("dropout", 0.1)
    encoder_emb_dropout = encoder_cfg["embeddings"].get("dropout", encoder_dropout)
    assert encoder_cfg["embeddings"]["embedding_dim"] == encoder_cfg["model_dim"], (
        "for transformer embedding_dim must be equal to model dim/hidden_size")
    
    encoder = TransformerEncoder(model_dim=encoder_cfg["model_dim"],ff_dim=encoder_cfg["ff_dim"],
                                num_layers=encoder_cfg["num_layers"],head_count=encoder_cfg["head_count"],
                                dropout=encoder_dropout, emb_dropout=encoder_emb_dropout,
                                layer_norm_position=encoder_cfg["layer_norm_position"],
                                src_pos_emb=encoder_cfg["src_pos_emb"],
                                max_src_len=encoder_cfg["max_src_len"],freeze=encoder_cfg["freeze"],
                                max_relative_position=encoder_cfg["max_relative_position"],
                                use_negative_distance=encoder_cfg["use_negative_distance"])
    # encoder = codebert_encode
    assert encoder
    
    # Build decoder
    decoder_dropout = decoder_cfg.get("dropout", 0.1)
    decoder_emb_dropout = decoder_cfg["embeddings"].get("dropout", decoder_dropout)
    decoder = TransformerDecoder(model_dim=decoder_cfg["model_dim"],ff_dim=decoder_cfg["ff_dim"],
                                num_layers=decoder_cfg["num_layers"],head_count=decoder_cfg["head_count"],
                                dropout=decoder_dropout, emb_dropout=decoder_emb_dropout,
                                layer_norm_position=decoder_cfg["layer_norm_position"],
                                trg_pos_emb=decoder_cfg["trg_pos_emb"],
                                max_trg_len=decoder_cfg["max_trg_len"],freeze=decoder_cfg["freeze"],
                                max_relative_positon=decoder_cfg["max_relative_position"],
                                use_negative_distance=decoder_cfg["use_negative_distance"])
    
    model = Transformer(encoder=encoder, decoder=decoder,
                        src_embed=src_embed, trg_embed=trg_embed,
                        src_vocab=src_vocab, trg_vocab=trg_vocab)
    
    # tie softmax layer with trg embeddings
    if model_cfg.get("tied_softmax", False):
        if trg_embed.lut.weight.shape == model.output_layer.weight.shape:
            # share trg embeddings and softmax layer:
            model.output_layer.weight = trg_embed.lut.weight
        else:
            raise ConfigurationError("For tied softmax, decoder embedding_dim == model dim.")
    
    # Custom Initialization of model parameters.
    # NOTE NO Initialization is better.
    # Initialize_model(model, model_cfg, src_pad_index, trg_pad_index)

    # Initializate embeddings from pre-trained embedding file.
    encoder_embed_path = encoder_cfg["embeddings"].get("load_pretrained", None)
    decoder_embed_path = decoder_cfg["embeddings"].get("load_pretrained", None)
    if encoder_embed_path and src_vocab is not None:
        logger.info("Loading pretrained src embedding...")
        model.src_embed.load_from_file(Path(encoder_embed_path), src_vocab)
    if decoder_embed_path and trg_vocab is not None and not model_cfg.get("tied_embeddings", False):
        logger.info("Loading pretrained trg embedding...")
        model.trg_embed.load_from_file(Path(decoder_embed_path), trg_vocab)
    
    logger.info("Transformer model is built.")
    # logger.info(model)
    model.log_parameters_list()
    return model

def Initialize_model(model: nn.Module, model_cfg:dict,
                    src_pad_index: int, trg_pad_index: int) -> None:
    """
    All initializer configuration is part of the 'model' section of the configuration file.
    The main initializer is set using the 'initializer' key.
        possible values are 'xavier', 'uniform', 'normal' or 'zeros'('xavier' is the default)
    when an initializer is set to 'uniform',
        then 'init_weight' sets the range for the values (-init_weight, init_weight)
    when an initializer is set to 'normal',
        then 'init_weight' sets the standard deviation for the weights (with mean 0).
    """
    init = model_cfg.get("initializer", "xavier_uniform")
    init_gain = float(model_cfg.get("init_gain", 1.0)) # for xavier
    init_weight = float(model_cfg.get("init_weight", 0.01)) # for uniform and normal

    embed_init = model_cfg.get("embed_initializer", "xavier_uniform")
    embed_init_gain = float(model_cfg.get("embed_init_gain",1.0))
    embed_init_weight = float(model_cfg.get("embed_init_weight", 0.01))

    def parse_init(init:str, weight: float, gain:float):
        weight = float(weight)
        assert weight > 0.0, "Incorrect init weight"
        if init.lower() == "xavier_uniform":
            return lambda p: nn.init.xavier_uniform_(p, gain=gain)
        elif init.lower() == "xavier_normal":
            return lambda p: nn.init.xavier_normal_(p, gain=gain)
        elif init.lower() == "uniform":
            return lambda p: nn.init.uniform_(p, a=-weight, b=weight)
        elif init.lower() == "normal":
            return lambda p: nn.init.normal_(p, mean=0.0, std=weight)
        elif init.lower() == "zeros":
            return lambda p: nn.init.zeros_(p)
    
    init_fn = parse_init(init=init, weight=init_weight, gain=init_gain)
    embed_init_fn = parse_init(init=embed_init, weight=embed_init_weight, gain=embed_init_gain)
    bias_init_fn = lambda p: nn.init.zeros_(p)

    with torch.no_grad():
        for name, param in model.named_parameters():
            if "emb" in name or "lut" in name:
                embed_init_fn(param)
            elif "bias" in name:
                bias_init_fn(param)
            elif len(param.size()) > 1:
                init_fn(param)

        # zero out paddings
        model.src_embed.lut.weight.data[src_pad_index].zero_()
        model.trg_embed.lut.weight.data[trg_pad_index].zero_()
        