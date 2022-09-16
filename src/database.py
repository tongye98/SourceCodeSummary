import numpy as np
from typing import Tuple
from src.faiss_index import FaissIndex

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
        if in_memory: # FIXME
            self.embeddings = np.load(embedding_path)
        else:
            self.embeddings = np.load(embedding_path, mmap_mode="r")

    def enhanced_search(self, embeddings:np.ndarray, top_k:int=16, retrieval_dropout:bool=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Search nearest top_k embeddings from Faiss index.
        embeddings: np.ndarray (batch_size, d)
        return distances np.ndarray (batch_size, top_k)
        return token_indices: np.ndarray (batch_size, top_k)
        return hidden: np.ndarray (batch_size, top_k, d)
        """
        if retrieval_dropout:
            distances, indices = self.index.search(embeddings, top_k+1)
            distances = distances[:, 1:]
            indices = indices[:, 1:]
        else:
            distances, indices = self.index.search(embeddings, top_k)

        token_indices = self.token_map[indices]
        batch_size = indices.shape[0]
        indices = indices.reshape(-1)
        hidden = self.embeddings[indices]
        d = hidden.shape[-1]
        hidden = hidden.reshape(batch_size, top_k, d)
        return distances, token_indices, hidden