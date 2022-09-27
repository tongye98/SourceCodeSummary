import faiss 
import numpy as np 
from typing import Tuple 
import re 

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
        else:
            index = faiss.index_factory(dimension, template, faiss.METRIC_INNER_PRODUCT)
        
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
        self.index.nprobe = nprobe

    @property
    def total(self):
        return self.index.ntotal