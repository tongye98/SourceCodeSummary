from hashlib import md5
import torch 
import logging
from pathlib import Path
from src.constants import BOS_ID
from src.datasets import BaseDataset
from src.helps import load_config, load_model_checkpoint
from src.datas import load_data, make_data_iter
from src.model import Transformer, build_model
from src.faiss_index import FaissIndex
from npy_append_array import NpyAppendArray
import tqdm
import numpy as np
from src.helps import make_logger

logger = logging.getLogger(__name__)

def getmd5(sequence: list) -> str:
    sequence = str(sequence)
    return md5(sequence.encode()).hexdigest()

def store_examples(model: Transformer, hidden_representation_path:str, token_map_path:str, data:BaseDataset, batch_size:int,
                    batch_type:str, seed:int, shuffle:bool, num_workers:int, device:torch.device) -> None:
    """
    Extract hidden states generted by trained model
    """
    data_iter = make_data_iter(dataset=data, sampler_seed=seed, shuffle=shuffle, batch_type=batch_type,
                                batch_size=batch_size, num_workers=num_workers)

    # Create Numpy NPY files by appending on the zero axis.
    npaa = NpyAppendArray(hidden_representation_path)
    # token_map file FIXME 
    token_map_file = open(token_map_path, "w", encoding="utf-8")
    already_meet_log = set()
    total_tokens = 0
    total_sequence = 0
    total_original_tokens = 0

    model.eval()
    # don't track gradients during validation
    with torch.no_grad():
         for batch_data in tqdm.tqdm(data_iter, desc="Store"):
            batch_data.move2cuda(device)
            src_input = batch_data.src
            trg_input = batch_data.trg_input
            src_mask = batch_data.src_mask
            trg_mask = batch_data.trg_mask
            trg_truth = batch_data.trg_truth
            trg_lengths = batch_data.trg_lengths
            total_sequence += batch_data.nseqs
            _, penultimate_representation, _ = model(return_type='encode_decode', src_input=src_input, 
                                        trg_input=trg_input, src_mask=src_mask, trg_mask=trg_mask)
            penultimate_representation = penultimate_representation.cpu().numpy().astype(np.float16)
            
            for i in range(batch_data.nseqs):
                # for each sentence
                total_original_tokens += trg_lengths[i]
                trg_tokens_id = trg_truth[i][0:trg_lengths[i]]   # FIXME include final <eos> token id(3)
                hidden_states = penultimate_representation[i][0:trg_lengths[i]]
                sequence = src_input[i].cpu().numpy().tolist() + [BOS_ID] # token id list

                for token_id, hidden_state in zip(trg_tokens_id, hidden_states):
                    meet_log = getmd5(sequence)
                    if meet_log in already_meet_log:
                        continue
                    else:
                        already_meet_log.add(meet_log)
                    npaa.append(hidden_state[np.newaxis,:])
                    token_map_file.write(f"{token_id}\n")
                    sequence += [token_id]
                    total_tokens += 1
            
    del npaa
    token_map_file.close()
    logger.info("Storing hidden state ended!")
    logger.info("Save {} sentences with {} tokens. | Original has {} tokens.".format(total_sequence, total_tokens, total_original_tokens))

def build_database(cfg_file: str, division:str, ckpt: str, hidden_representation_path:str, token_map_path:str, index_path:str):
    """
    The function to store hidden states generated from trained transformer model.
    Handles loading a model from checkpoint, generating hidden states by force decoding and storing them.
    division: which dataset to build database.
    ckpt: use pre-trained model to produce representation.
    hidden_representation_path: where to store token hidden representation.
    token_map_path: where to store corresponding token_map
    index_path: where to store FAISS Index.
    """
    # make logger
    # FIXME this is just for test
    model_dir = Path("test/")
    make_logger(model_dir, mode="build_database")

    logger.info("Load config...")
    cfg = load_config(cfg_file)
    use_cuda = cfg["training"]["use_cuda"]
    device = torch.device("cuda" if use_cuda else "cpu")
    batch_size = cfg["training"]["batch_size"]
    batch_type = cfg["training"]["batch_type"]
    seed = cfg["training"]["random_seed"]
    shuffle= cfg["training"]["shuffle"]
    num_workers = cfg["training"]["num_workers"]

    assert division in ["train", "dev", "test"]
    logger.info("devision = {}".format(division))

    assert ckpt is not None
    # load model checkpoint
    logger.info("Load model checkpoint")
    model_checkpoint = load_model_checkpoint(Path(ckpt), device)

    # load data
    logger.info("Load data")
    train_data, dev_data, test_data, src_vocab, trg_vocab = load_data(data_cfg=cfg["data"])

    # build model and load parameters into it
    model = build_model(model_cfg=cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)
    model.load_state_dict(model_checkpoint["model_state"])

    if use_cuda:
        model.to(device)

    if division == "train":
        logger.info("Store train examples...")
        store_examples(model, hidden_representation_path=hidden_representation_path, token_map_path=token_map_path,
                       data=train_data, batch_size=batch_size, batch_type=batch_type, seed=seed,
                       shuffle=shuffle, num_workers=num_workers, device=device)

        logger.info("train index...")
        index = FaissIndex()
        index.train(hidden_representation_path)
        index.add(hidden_representation_path)
        index.export(index_path)
        del index

    return None