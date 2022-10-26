"""
build_database module.
FaissIndex, Database, EnhancedDatabase, build_database.
"""
import logging
import torch 
import tqdm 
import faiss 
import re
import numpy as np
from pathlib import Path
from typing import Tuple 
from hashlib import md5
from src.constants import BOS_ID
from src.datasets import BaseDataset
from src.helps import load_config, load_model_checkpoint, make_logger
from src.datas import load_data, make_data_iter
from src.model import Transformer, build_model
from npy_append_array import NpyAppendArray
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class FaissIndex(object):
    """
    FaissIndex class. factory_template; index_type
    For train index (core: self.index)
    """
    def __init__(self, factory_template:str="IVF256,PQ32", load_index_path:str=None,
                 use_gpu:bool=True, index_type:str="L2") -> None:
        super().__init__()
        self.factory_template = factory_template
        self.gpu_num = faiss.get_num_gpus()
        self.use_gpu = use_gpu and (self.gpu_num > 0)
        self.index_type= index_type
        self._is_trained= False
        if load_index_path != None:
            self.load(index_path=load_index_path)
        
    def load(self, index_path:str) -> faiss.Index:
        self.index = faiss.read_index(index_path)
        if self.use_gpu:
            self.index = faiss.index_cpu_to_all_gpus(self.index)
        self._is_trained = True
    
    @property
    def is_trained(self) -> bool:
        return self._is_trained
    
    def train(self, hidden_representation_path:str) -> None:
        embeddings = np.load(hidden_representation_path, mmap_mode="r")
        total_samples, dimension = embeddings.shape
        logger.info("total samples = {}, dimension = {}".format(total_samples, dimension))
        del embeddings
        centroids, training_samples = self._get_clustering_parameters(total_samples)
        self.index = self._initialize_index(dimension, centroids)
        training_embeddinigs = self._get_training_embeddings(hidden_representation_path, training_samples).astype(np.float32)
        self.index.train(training_embeddinigs)
        self._is_trained = True

    def _get_clustering_parameters(self, total_samples: int) -> Tuple[int, int]:
        if 0 < total_samples <= 10 ** 6:
            centroids = int(8 * total_samples ** 0.5)
            training_samples = total_samples
        elif 10 ** 6 < total_samples <= 10 ** 7:
            centroids = 65536
            training_samples = min(total_samples, 64 * centroids)
        else:
            centroids = 262144
            training_samples = min(total_samples, 64 * centroids)
        return centroids, training_samples
    
    def _initialize_index(self, dimension:int, centroids:int) -> faiss.Index:
        template = re.compile(r"IVF\d*").sub(f"IVF{centroids}", self.factory_template)
        if self.index_type == "L2":
            index = faiss.index_factory(dimension, template, faiss.METRIC_L2)
        elif self.index_type == "INNER":
            index = faiss.index_factory(dimension, template, faiss.METRIC_INNER_PRODUCT)
        else:
            assert False, "Double check index_type!"
        
        if self.use_gpu:
            index = faiss.index_cpu_to_all_gpus(index)
        
        return index
    
    def _get_training_embeddings(self, embeddings_path:str, training_samples: int) -> np.ndarray:
        embeddings = np.load(embeddings_path, mmap_mode="r")
        total_samples = embeddings.shape[0]
        sample_indices = np.random.choice(total_samples, training_samples, replace=False)
        sample_indices.sort()
        training_embeddings = embeddings[sample_indices]
        return training_embeddings        
    
    def add(self, hidden_representation_path: str, batch_size: int = 10000) -> None:
        assert self.is_trained
        embeddings = np.load(hidden_representation_path)
        total_samples = embeddings.shape[0]
        for i in range(0, total_samples, batch_size):
            start = i 
            end = min(total_samples, i+batch_size)
            batch_embeddings = embeddings[start: end].astype(np.float32)
            self.index.add(batch_embeddings)
        del embeddings
    
    def export(self, index_path:str) -> None:
        assert self.is_trained
        if self.use_gpu:
            index = faiss.index_gpu_to_cpu(self.index)
        else:
            index = self.index 
        faiss.write_index(index, index_path)
    
    def search(self, embeddings: np.ndarray, top_k:int=1)-> Tuple[np.ndarray, np.ndarray]:
        assert self.is_trained
        distances, indices = self.index.search(embeddings, k=top_k)
        return distances, indices

    def set_prob(self, nprobe):
        # default nprobe = 1, can try a few more
        # nprobe: search in how many cluster, defualt:1; the bigger nprobe, the result is more accurate, but speed is lower
        self.index.nprobe = nprobe

    @property
    def total(self):
        return self.index.ntotal

class Database(object):
    """
    Initilize with index_path, which is built offline,
    and token path which mapping retrieval indices to token id.
    """
    def __init__(self, index_path:str, token_map_path: str, nprobe:int=16) -> None:
        super().__init__()
        self.index = FaissIndex(load_index_path=index_path, use_gpu=True)
        self.index.set_prob(nprobe)
        self.token_map = self.load_token_mapping(token_map_path)
    
    @staticmethod
    def load_token_mapping(token_map_path: str) -> np.ndarray:
        """
        Load token mapping from file.
        """
        with open(token_map_path) as f:
            token_map = [int(token_id) for token_id in f.readlines()]
        token_map = np.asarray(token_map).astype(np.int32)
        return token_map
    
    def search(self, embeddings:np.ndarray, top_k: int=16) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search nearest top_k embeddings from the Faiss index.
        embeddings: np.ndarray (batch_size, d)
        return token_indices: np.ndarray (batch_size, top_k)
        """
        distances, indices = self.index.search(embeddings, top_k)
        token_indices = self.token_map[indices]
        return distances, token_indices

class EnhancedDatabase(Database):
    def __init__(self, index_path:str, token_map_path:str, embedding_path:str, nprobe:int=16, in_memory:bool=True) -> None:
        super().__init__(index_path, token_map_path, nprobe)
        if in_memory: # load data to memory
            self.embeddings = np.load(embedding_path)
        else:         # the data still on disk
            self.embeddings = np.load(embedding_path, mmap_mode="r")

    def enhanced_search(self, hidden:np.ndarray, top_k:int=8, retrieval_dropout:bool=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Search nearest top_k embeddings from Faiss index.
        hidden: np.ndarray [batch_size*trg_len, model_dim]
        return distances np.ndarray (batch_size*trg_len, top_k)
        return token_indices: np.ndarray (batch_size*trg_len, top_k)
        return searched_hidden: np.ndarray (batch_size*trg_len, top_k, model_dim)
        """
        if retrieval_dropout:
            distances, indices = self.index.search(hidden, top_k + 1)
            distances = distances[:, 1:]
            indices = indices[:, 1:]
        else:
            distances, indices = self.index.search(hidden, top_k)
        # distances [batch_size*trg_len, top_k]
        # indices [batch_size*trg_len, top_k]

        token_indices = self.token_map[indices]
        # token_indices [batch_size*trg_len, top_k]
        searched_hidden = self.embeddings[indices]
        # searched_hidden [batch_size*trg_len, top_k, dim]
        return distances, token_indices, searched_hidden

def getmd5(sequence: list) -> str:
    sequence = str(sequence)
    return md5(sequence.encode()).hexdigest()

def store_examples(model: Transformer, hidden_representation_path:str, token_map_path:str, data:BaseDataset, batch_size:int,
                    batch_type:str, seed:int, shuffle:bool, num_workers:int, device:torch.device, data_dtype: str) -> None:
    """
    Extract hidden states generted by trained model
    """
    data_iter = make_data_iter(dataset=data, sampler_seed=seed, shuffle=shuffle, batch_type=batch_type,
                                batch_size=batch_size, num_workers=num_workers)

    # Create Numpy NPY files by appending on the zero axis.
    npaa = NpyAppendArray(hidden_representation_path)
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
            decode_output, penultimate_representation, _ = model(return_type='encode_decode', src_input=src_input, 
                                        trg_input=trg_input, src_mask=src_mask, trg_mask=trg_mask)
            penultimate_representation = penultimate_representation.cpu().numpy().astype(data_dtype)
            
            for i in range(batch_data.nseqs):
                # for each sentence
                total_original_tokens += trg_lengths[i]
                trg_tokens_id = trg_truth[i][0:trg_lengths[i]]   #include final <eos> token id(3)
                hidden_states = penultimate_representation[i][0:trg_lengths[i]]
                # faiss.normalize_L2(hidden_states)
                sequence = src_input[i].cpu().numpy().tolist() + [BOS_ID] # token id list

                for token_id, hidden_state in zip(trg_tokens_id, hidden_states):
                    # meet_log = getmd5(sequence)
                    # if meet_log in already_meet_log:
                    #     logger.info(sequence)
                    #     continue
                    # else:
                    #     already_meet_log.add(meet_log)
                    npaa.append(hidden_state[np.newaxis,:])
                    token_map_file.write(f"{token_id}\n")
                    sequence += [token_id]
                    total_tokens += 1
            
    del npaa
    token_map_file.close()
    logger.info("Storing hidden state ended!")
    logger.info("Save {} sentences with {} tokens. | Original has {} tokens.".format(total_sequence, total_tokens, total_original_tokens))

def build_database(cfg_file: str, division:str, ckpt: str, hidden_representation_path:str, 
                    token_map_path:str, index_path:str, data_dtype:str):
    """
    The function to store hidden states generated from trained transformer model.
    Handles loading a model from checkpoint, generating hidden states by force decoding and storing them.
    division: which dataset to build database.
    ckpt: use pre-trained model to produce representation.
    hidden_representation_path: where to store token hidden representation.
    token_map_path: where to store corresponding token_map
    index_path: where to store FAISS Index.
    """
    cfg = load_config(Path(cfg_file))
    model_dir = cfg["training"].get("model_dir", None)

    make_logger(Path(model_dir), mode="build_database")

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
                       shuffle=shuffle, num_workers=num_workers, device=device, data_dtype=data_dtype)
        logger.info("Store train examples done!")

        logger.info("train index...")
        index = FaissIndex(index_type="L2")
        index.train(hidden_representation_path)
        index.add(hidden_representation_path)
        index.export(index_path)
        logger.info("train index done!")

        del index
