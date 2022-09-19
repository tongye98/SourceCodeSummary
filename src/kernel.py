import torch 
from typing import Tuple, Union 

class Kernel(object):
    def __init__(self) -> None:
        super().__init__()
    
    def similarity(self, distances:torch.Tensor, bandwidth:Union[float, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError
    
    def compute_example_based_distribution(self, distances:torch.Tensor, bandwidth:Union[float, torch.Tensor], 
                                            token_indices:torch.Tensor, vocab_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.similarity(distances, bandwidth)
        # distances[batch_size*trg_len, top_k]
        sparse_distribution = torch.softmax(scores, dim=-1)
        # sparse_distribution [batch_size*trg_len, top_k]        
        zeros = torch.zeros(size=(sparse_distribution.size(0), vocab_size), device=sparse_distribution.device, dtype=sparse_distribution.dtype)
        distribution = torch.scatter_add(zeros, -1, token_indices, sparse_distribution)
        # distribution [batch_size*trg_len, vocab_size]
        return distribution, sparse_distribution

class GaussianKernel(Kernel):
    def __init__(self) -> None:
        super().__init__()
    
    def similarity(self, distances: torch.Tensor, bandwidth: Union[float, torch.Tensor]) -> torch.Tensor:
        return - distances / bandwidth

class LaplacianKernel(Kernel):
    def __init__(self) -> None:
        super().__init__()
    
    def similarity(self, distances: torch.Tensor, bandwidth: Union[float, torch.Tensor]) -> torch.Tensor:
        return - torch.sqrt(distances) / bandwidth