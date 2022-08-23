from distutils.log import error
from msilib.schema import Error
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.vocabulary import Vocabulary
from src.embeddings import Embeddings
from src.encoders import TransformerEncoder
from src.decoders import TransformerDecoder 
from src.helps import ConfigurationError
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class Transformer(nn.Module):
    """
    Transformer Model
    """
    super().__init__()



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
                                freeze=encoder_cfg["freeze"])
    
    # Build decoder
    decoder_dropout = decoder_cfg.get("dropout", 0.1)
    decoder_emb_dropout = decoder_cfg["embeddings"].get("dropout",decoder_dropout)
    decoder = TransformerDecoder(model_dim=decoder_cfg["model_dim"],ff_dim=decoder_cfg["ff_dim"],
                                num_layers=decoder_cfg["num_layers"],head_count=decoder_cfg["head_count"],
                                dropout=decoder_dropout, emb_dropout=decoder_emb_dropout,
                                layer_norm_position=decoder_cfg["layer_norm_position"],
                                freeze=decoder_cfg["freeze"])
    
    model = Transformer(encoder=encoder, decoder=decoder,
                        src_embed=src_embed, trg_embed=trg_embed,
                        src_vocab=src_vocab, trg_vocab=trg_vocab)
    
    # tie softmax layer with trg embeddings
    if model_cfg.get("tied_softmax", False):
        if trg_embed.lut.weight.shape == model.decoder.output_layer.weight.shape:
            # share trg embeddings and softmax layer:
            model.decoder.output_layer.weight = trg_embed.lut.weight
        else:
            raise ConfigurationError("For tied softmax, decoder embedding_dim == decoder hidden_size.")
    
    # Custom Initialization of model parameters.
    # TODO

    # Initializate embeddings from pre-trained embedding file.
    encoder_embed_path = encoder_cfg["embeddings"].get("load_pretrained",None)
    decoder_embed_path = decoder_cfg["embeddings"].get("load_pretrained",None)
    if encoder_embed_path and src_vocab is not None:
        logger.info("Loading pretrained src embedding...")
        model.src_embed.load_from_file(Path(encoder_embed_path), src_vocab)
    if decoder_embed_path and trg_vocab is not None and not model_cfg.get("tied_embeddings",False):
        logger.info("Loading pretrained trg embedding...")
        model.trg_embed.load_from_file(Path(decoder_embed_path), trg_vocab)
    
    logger.info("Transformer model built.")
    return model
