"""
build code semantic database module.
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
from src.datasets import BaseDataset
from src.helps import load_config, load_model_checkpoint, make_logger
from src.datas import load_data, make_data_iter
from src.model import Transformer, build_model
from npy_append_array import NpyAppendArray

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
        assert self.index_type in {"L2", "INNER"}
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
        # centroids, training_samples = self._get_clustering_parameters(total_samples)
        self.index = self.our_initialize_index(dimension)
        training_embeddinigs = self._get_training_embeddings(hidden_representation_path, total_samples).astype(np.float32)
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
    
    def our_initialize_index(self, dimension) -> faiss.Index:
        if self.index_type == "L2":
            index = faiss.index_factory(dimension, "Flat", faiss.METRIC_L2)
        elif self.index_type == "INNER":
            index = faiss.index_factory(dimension, "Flat", faiss.METRIC_INNER_PRODUCT)

        if self.use_gpu:
            index = faiss.index_cpu_to_all_gpus(index)

        return index

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
        if self.index_type == "INNER":
            faiss.normalize_L2(training_embeddings)
        return training_embeddings        
    
    def add(self, hidden_representation_path: str, batch_size: int = 10000) -> None:
        assert self.is_trained
        embeddings = np.load(hidden_representation_path)
        total_samples = embeddings.shape[0]
        for i in range(0, total_samples, batch_size):
            start = i 
            end = min(total_samples, i+batch_size)
            batch_embeddings = embeddings[start: end].astype(np.float32)
            if self.index_type == "INNER":
                faiss.normalize_L2(batch_embeddings)
            self.index.add(batch_embeddings)
        del embeddings
    
    def export(self, index_path:str) -> None:
        assert self.is_trained
        if self.use_gpu:
            index = faiss.index_gpu_to_cpu(self.index)
        else:
            index = self.index 
        faiss.write_index(index, index_path)
    
    def search(self, embeddings: np.ndarray, top_k:int=8)-> Tuple[np.ndarray, np.ndarray]:
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
    def __init__(self, index_path:str, token_map_path: str, index_type: str, nprobe:int=16) -> None:
        super().__init__()
        self.index = FaissIndex(load_index_path=index_path, use_gpu=True, index_type=index_type)
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
        return distances: np.ndarray
        """
        if self.index.index_type == "INNER":
            faiss.normalize_L2(embeddings)
        distances, indices = self.index.search(embeddings, top_k)
        token_indices = self.token_map[indices]
        return distances, token_indices

class EnhancedDatabase(Database):
    def __init__(self, index_path:str, token_map_path:str, embedding_path:str, index_type:str, nprobe:int=16, in_memory:bool=True) -> None:
        super().__init__(index_path, token_map_path, index_type, nprobe)
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

        token_indices = self.token_map[indices]         # token_indices [batch_size*trg_len, top_k]
        searched_hidden = self.embeddings[indices]      # searched_hidden [batch_size*trg_len, top_k, dim]
        return distances, token_indices, searched_hidden

def getmd5(sequence: list) -> str:
    sequence = str(sequence)
    return md5(sequence.encode()).hexdigest()

def store_code_semantic(model: Transformer, code_semantic_path:str,
                   data:BaseDataset, batch_size:int, batch_type:str, seed:int, shuffle:bool,
                   num_workers:int, device:torch.device):
    """
    Extract hidden states generated by trained model.
    """
    data_iter = make_data_iter(dataset=data, sampler_seed=seed, shuffle=False, batch_type=batch_type,
                                batch_size=batch_size, num_workers=num_workers)

    # Create Numpy NPY files by appending on the zero axis.
    npaa = NpyAppendArray(code_semantic_path)
    #token_map_file = open(token_map_path, "w", encoding="utf-8")

    # Statistic Analysis
    total_tokens = 0
    total_sequence = 0
    total_original_tokens = 0

    # don't track gradients during validation
    model.eval()
    with torch.no_grad():
        for batch_data in tqdm.tqdm(data_iter, desc="Store"):
            batch_data.move2cuda(device)
            src_input = batch_data.src
            src_lengths = batch_data.src_lengths
            trg_input = batch_data.trg_input
            src_mask = batch_data.src_mask  # shape [batch_size, 1, pad_src_length]
            trg_mask = batch_data.trg_mask
            trg_truth = batch_data.trg_truth
            trg_lengths = batch_data.trg_lengths
            total_sequence += batch_data.nseqs

            encode_output = model(return_type="encode", src_input=src_input, trg_input=trg_input, src_mask=src_mask, trg_mask=trg_mask)
            # encode_output [batch_size, src_len, model_dim]

            for i in range(batch_data.nseqs):
                sentence_src_length = src_lengths[i]
                sentence_encode_output = encode_output[i] # [src_len, model_dim]
                sentence_encode_output_nopad = sentence_encode_output[:sentence_src_length] # [real_src_len, model_dim]
                sentence_encode_semantic = torch.mean(sentence_encode_output_nopad, dim=0) # [model_dim]
                sentence_encode_semantic = sentence_encode_semantic.cpu().numpy().astype("float32")
                npaa.append(sentence_encode_semantic[np.newaxis,:])
            
    del npaa

def build_code_semantic_database(cfg_file: str):
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
    model_dir = cfg["retriever"].get("code_semantic_dir", None)
    make_logger(Path(model_dir), mode="test_index3")

    use_cuda = cfg["training"]["use_cuda"]
    device = torch.device("cuda" if use_cuda else "cpu")
    batch_size = cfg["training"]["batch_size"]
    batch_type = cfg["training"]["batch_type"]
    seed = cfg["training"]["random_seed"]
    shuffle= cfg["training"]["shuffle"]
    num_workers = cfg["training"]["num_workers"]

    # load data
    logger.info("Load data")
    train_data, dev_data, test_data, src_vocab, trg_vocab = load_data(data_cfg=cfg["data"])

    # load model checkpoint
    logger.info("Load model checkpoint")
    ckpt = cfg["retriever"]["pre_trained_model_path"]
    model_checkpoint = load_model_checkpoint(Path(ckpt), device)

    # build model and load parameters into it
    model = build_model(model_cfg=cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)
    model.load_state_dict(model_checkpoint["model_state"])

    if use_cuda:
        model.to(device)

    # logger.info("Store train examples code semantic...")
    # code_semantic_path = cfg["retriever"]["code_semantic_path"]
    # # use_code_representation = cfg["retriever"]["use_code_representation"]
    # store_code_semantic(model, code_semantic_path=code_semantic_path,
    #                 data=train_data, batch_size=batch_size, batch_type=batch_type, seed=seed,
    #                 shuffle=shuffle, num_workers=num_workers, device=device)
    # logger.info("Store train examples code semantic done!")

    # logger.info("train index...")
    code_index_path = cfg["retriever"]["code_index_path"]
    # code_semantic_path = cfg["retriever"]["code_semantic_path"]
    index_type = cfg["retriever"]["index_type"]
    # index = FaissIndex(index_type=index_type)
    # index.train(code_semantic_path)
    # index.add(code_semantic_path)
    # index.export(code_index_path)
    # logger.info("train index done!")

    # del index


    
    index = FaissIndex(load_index_path=code_index_path, use_gpu=True, index_type=index_type)
    index.set_prob(16)
    data_iter = make_data_iter(dataset=test_data, sampler_seed=seed, shuffle=False, batch_type=batch_type,
                    batch_size=batch_size, num_workers=num_workers)
    
    model.eval()
    with torch.no_grad():
        batch_encode_semantic_distances= []
        batch_encode_similar_indices = []
        for batch_data in tqdm.tqdm(data_iter, desc="Test index"):
            batch_data.move2cuda(device)
            src_input = batch_data.src
            src_lengths = batch_data.src_lengths
            trg_input = batch_data.trg_input
            src_mask = batch_data.src_mask  # shape [batch_size, 1, pad_src_length]
            trg_mask = batch_data.trg_mask
            trg_truth = batch_data.trg_truth
            trg_lengths = batch_data.trg_lengths

            encode_output = model(return_type="encode", src_input=src_input, trg_input=trg_input, src_mask=src_mask, trg_mask=trg_mask)
            # encode_output [batch_size, src_len, model_dim]

            batch_encode_semantic = []
            for i in range(batch_data.nseqs):
                sentence_src_length = src_lengths[i]
                sentence_encode_output = encode_output[i] # [src_len, model_dim]
                sentence_encode_output_nopad = sentence_encode_output[:sentence_src_length] # [real_src_len, model_dim]
                sentence_encode_semantic = torch.mean(sentence_encode_output_nopad, dim=0) # [model_dim]

                # sentence_encode_semantic = sentence_encode_semantic.cpu().numpy().astype("float32")
                # sentence_encode_semantic = sentence_encode_semantic[np.newaxis,:]
                batch_encode_semantic.append(sentence_encode_semantic.unsqueeze(0))
            
            batch_encode_semantic = torch.cat(batch_encode_semantic, dim=0)
            # batch_encode_semantic [batch_size, model_dim]
            batch_encode_semantic = batch_encode_semantic.cpu().numpy().astype("float32")

            if index_type == "INNER":
                faiss.normalize_L2(batch_encode_semantic)
            
            topk = 4
            distances, indices = index.search(batch_encode_semantic, top_k=topk)
            # logger.info("distances = {}".format(distances))
            # logger.info("indices = {}".format(indices))
            batch_encode_semantic_distances.append(distances)
            batch_encode_similar_indices.append(indices)
    
    whole_distances = np.concatenate(batch_encode_semantic_distances, axis=0)
    whole_indices = np.concatenate(batch_encode_similar_indices, axis=0)
    logger.info("whole distances shape = {}".format(whole_distances.shape)) # [21745, 4]
    logger.info("whole indices shape = {}".format(whole_indices.shape)) # [21745, 4]

    distances = whole_distances[:, 2]
    indices = whole_indices[:, 2]

    distance_map_file = open("distance3_map_file", "w", encoding="utf-8")
    indice_map_file = open("indice3_map_file","w",encoding="utf-8")

    for distance, indice in zip(distances, indices):
        distance_map_file.write(f"{distance}\n")
        indice_map_file.write(f"{indice}\n")
    
    distance_map_file.close()
    indice_map_file.close()




