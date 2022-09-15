import numpy as np
from src.faiss_index import FaissIndex

class Database(object):
    """
    Initilize with index_path, which is built offline,
    and token path which mapping retrieval indices to token id.
    """
    def __init__(self, index_path, token_map_path: str, nprobe:int=16) -> None:
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
    
    