import logging
import math
import torch 
from torch import Tensor, nn
import torch.nn.functional as F
from typing import Dict
from pathlib import Path
import numpy as np
from src.transformer_layers import CopyGenerator, GlobalAttention
from src.vocabulary import Vocabulary
from src.encoders import TransformerEncoder
from src.decoders import TransformerDecoder 
from src.helps import ConfigurationError, freeze_params 
from src.initialization import Initialize_model
from src.loss import XentLoss, CopyGeneratorLoss
from src.retriever import Retriever

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

class Transformer(nn.Module):
    """
    Transformer Model
    """
    def __init__(self, encoder: TransformerEncoder, decoder:TransformerDecoder,
                 src_embed: Embeddings, trg_embed: Embeddings,
                 src_vocab: Vocabulary, trg_vocab: Vocabulary, 
                 copy: bool=False) -> None:
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
        self.output_layer = nn.Linear(self.model_dim, self.trg_vocab_size, bias=False)

        self.loss_function = None
        self.copy = copy
        if self.copy:
            self.copy_attention_score = GlobalAttention(self.model_dim)
            self.copy_generator = CopyGenerator(self.model_dim, self.trg_vocab, self.output_layer)
            self.loss_function = CopyGeneratorLoss(self.trg_vocab_size, force_copy=False)
        else:
            self.loss_function = XentLoss(pad_index=self.pad_index, smoothing=0)

    def forward(self, return_type: str=None,
                src_input:Tensor=None, trg_input:Tensor=None,
                src_mask:Tensor=None, trg_mask:Tensor=None,
                encoder_output: Tensor=None, trg_truth:Tensor=None,
                copy_param=None):
        """
        return_type: one of {"loss", "encode", "decode"}
        src_input [batch_size, src_len]
        trg_input [batch_size, trg_len]
        src_mask [batch_size, 1, src_len]
        trg_mask [batch_size, trg_len, trg_len]
        encoder_output [batch_size, src_len, model_dim]
        """
        if return_type is None:
            raise ValueError("Please specify return_type: {'loss','encode','decode'}")
        if return_type == "loss":
            assert self.loss_function is not None
            assert trg_input is not None and trg_mask is not None
            encode_output = self.encode(src_input, src_mask)
            decode_output, _, cross_attention_weight = self.decode(trg_input, encode_output, src_mask, trg_mask)
            # decode_output [batch_size, trg_len, model_dim]
            if self.copy is False:
                logits = self.output_layer(decode_output)
                # logits [batch_size, trg_len, trg_vocab_size]
                log_probs = F.log_softmax(logits, dim=-1)
                batch_loss = self.loss_function(log_probs, target=trg_truth)
                # return batch loss = sum over all sentences of all tokens in the batch that are not pad
            else:
                # use copy mechanism
                assert copy_param is not None
                fuse_score, attention_score = self.copy_attention_score(decode_output, encode_output, src_mask)
                # attention_score [batch_size, trg_len, src_len]
                source_maps = copy_param["source_maps"]
                prob = self.copy_generator(decode_output, attention_score, source_maps)
                # prob [batch_size, trg_len, trg_vocab_size + extra_words]
                alignments = copy_param["alignments"]
                # FIXME
                assert False
                batch_loss = self.loss_function(prob, alignments, trg_truth)
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
            return decode_output, penultimate_representation, cross_attention_weight
        elif return_type == "retrieval_loss":
            assert self.loss_function is not None 
            assert self.retriever is not None 
            assert isinstance(self.retriever, Retriever)
            encode_output = self.encode(src_input, src_mask)
            decode_output, penultimate_representation, cross_attention_weight = self.decode(trg_input, encode_output, src_mask, trg_mask)
            logits = self.output_layer(decode_output)
            log_probs = self.retriever(hidden=penultimate_representation, logits=logits)
            # log_probs [batch_size, trg_len, vocab_size]
            batch_loss = self.loss_function(log_probs, target=trg_truth)
            return batch_loss
        else:
            raise ValueError("return_type must be one of {'loss','encode','decode'}")
    
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
                f"\tloss_function={self.loss_function},\n"
                f"\tcopy={self.copy})")
    
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

def build_model(model_cfg: dict=None,
                src_vocab: Vocabulary=None,
                trg_vocab: Vocabulary=None) -> Transformer:
    """
    Build and initialize the transformer according to the configuration.
    cfg: yaml model part.
    """
    logger.info("Building Transformer Model...")
    encoder_cfg = model_cfg["encoder"]
    decoder_cfg = model_cfg["decoder"]

    src_pad_index = src_vocab.pad_index
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
                        src_vocab=src_vocab, trg_vocab=trg_vocab,
                        copy=model_cfg["copy"])
    
    # tie softmax layer with trg embeddings
    if model_cfg.get("tied_softmax", False):
        if trg_embed.lut.weight.shape == model.output_layer.weight.shape:
            # share trg embeddings and softmax layer:
            model.output_layer.weight = trg_embed.lut.weight
        else:
            raise ConfigurationError("For tied softmax, decoder embedding_dim == model dim.")
    
    # Custom Initialization of model parameters.
    # FIXME initialize or scale problem?
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
    logger.info(model)
    model.log_parameters_list()
    return model