import torch 
import logging
from src.datasets import BaseDataset

from src.helps import load_config, load_model_checkpoint
from src.datas import load_data, make_data_iter
from src.model import Transformer, build_model
from src.faiss_index import FaissIndex
from npy_append_array import NpyAppendArray

logger = logging.getLogger(__name__)

def store_examples(model: Transformer, embedding_path:str, token_map_path:str, data:BaseDataset, batch_size:int,
                    batch_type:str, seed:int, shuffle:bool, num_workers:int, device:torch.device) -> None:
    """
    Extract hidden states generted by trained model
    """
    data_iter = make_data_iter(dataset=data, sampler_seed=seed, shuffle=shuffle, batch_type=batch_type,
                                batch_size=batch_size, num_workers=num_workers, device=device)

    npaa = NpyAppendArray(embedding_path)
    token_map_file = open(token_map_path, "w", encoding="utf-8")

    model.eval()
    with torch.no_grad():
         for batch_data in iter(data_iter):
            src_input = batch_data.src
            trg_input = batch_data.trg_input
            src_mask = batch_data.src_mask
            trg_mask = batch_data.trg_mask
            trg_truth = batch_data.trg
            trg_length = batch_data.trg_length
            _, penultimate_representation, _ = model(return_type='encode_decode', src_input=src_input, 
                                        trg_input=trg_input, src_mask=src_mask, trg_mask=trg_mask)
            
            for i in range(batch_data.nseqs):
                # for each sentence
                trg_tokens_id = trg_truth[i][0:trg_length[i]]
                hidden_states = penultimate_representation[i][0:trg_length[i]]
                for token_id, hidden_state in zip(trg_tokens_id, hidden_states):
                    # for each token
                    npaa.append(hidden_state)
                    token_map_file.write(f"{token_id}\n")
    
    del npaa
    token_map_file.close()

def build_database(cfg_file: str, division:str, ckpt: str):
    """
    The function to store hidden states generated from trained transformer model.
    Handles loading a model from checkpoint, generating hidden states by force decoding and storing them.
    """
    logger.info("Load config...")
    cfg = load_config(cfg_file)
    model_dir = cfg["training"]["model_dir"]

    assert division in ["train", "dev", "test"]
    logger.info("devision = {}".format(division))

    assert ckpt is not None
    # load model checkpoint
    logger.info("Load model checkpoint")
    model_checkpoint = load_model_checkpoint(ckpt, device)

    # load data
    logger.info("Load data")
    train_data, dev_data, test_data, src_vocab, trg_vocab = load_data(data_cfg=cfg["data"])

    # build model and load parameters into it
    model = build_model(model_cfg=cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)
    model.load_state_dict(model_checkpoint["model_state"])

    if use_cuda:
        model.to(device)
    
    if division == "train":
        logger.info("Store examples.")
        store_examples(model, embedding_path=embedding_path, token_map_path=token_map_path,
        data=train_data, batch_size=batch_size)

        logger.info("train index")
        index = FaissIndex()
        index.train(embedding_path)
        index.add(embedding_path)
        index.export(index_path)
        del index

    return None