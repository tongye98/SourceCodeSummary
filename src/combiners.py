from logging import logProcesses
from turtle import distance
from typing import Tuple
import torch 
import torch.nn as nn
from src.database import Database 
import math
import torch.nn.functional as F

class Combiner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, hidden:torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """
        hidden: [batch_size, seq_len, model_dim]
        logits: [batch_size, seq_len, model_dim]
        return:
            log_probs: [batch_size, seq_len, model_dim]
        """
        raise NotImplementedError("The forward method is not implemented in the Combiner class.")
    
    def detailed_forward(self, hidden:torch.Tensor, logits:torch.Tensor) -> Tuple[torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        hidden: [batch_size, seq_len, model_dim]
        logits: [batch_size, seq_len, model_dim]
        return:

        """
        raise NotImplementedError

class NoCombiner(Combiner):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, hidden:torch.Tensor, logits:torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs

class StaticCombiner(Combiner):
    def __init__(self, database:Database, top_k:int, mixing_weight:float, kernel:Kernel, bandwidth:float) -> None:
        super().__init__()
        self.database = database
        self.top_k = top_k 
        self.mixing_weight = mixing_weight
        self.kernel = kernel
        self.bandwidth = bandwidth
    
    def forward(self, hidden:torch.Tensor, logits:torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, model_dim = hidden.size()
        vocab_size = logits.size(-1)
        hidden = hidden.view(batch_size*seq_len, model_dim)
        logits = logits.view(batch_size*seq_len, vocab_size)

        model_based_distribution = F.softmax(logits, dim=-1)
        distances, token_indices = self.database.search(hidden.cpu().numpy(), top_k=self.top_k)
        distances = torch.FloatTensor(distances).to(hidden.device)
        token_indices = torch.LongTensor(token_indices).to(hidden.device)
        example_based_distribution, _ = self.kernel.compute_example_based_distribution(distance, self.bandwidth, token_indices, vocab_size)

        mixed_distribution = (1 - self.mixing_weight)*model_based_distribution + self.mixing_weight*example_based_distribution
        log_probs = torch.log(mixed_distribution)
        log_probs = log_probs.view(batch_size, seq_len, vocab_size).contiguous()
        return log_probs

class DynamicCombiner(Combiner):
    def __init__(self, database:Database, top_k:int, mixing_weight:float, kernel:Kernel, bandwidth:float) -> None:
        super().__init__()
        self.database =  database
        self.top_k = top_k 
        self.kernel = kernel
        dimension = database.index.index.d
        self.bandwidth_estimator = nn.Linear(2*dimension, 1)
        if isinstance(kernel, GaussianKernel):
            self.bandwidth_estimator.bias.data[0] = math.log(100)
        else:
            self.bandwidth_estimator.bias.data[0] = math.log(10)
        self.mixing_weight_estimator = nn.Sequential(nn.Linear(2*dimension, dimension),nn.ReLU(),nn.Linear(dimension,1))
    
    def forward(self, hidden:torch.Tensor, logits:torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, model_dim = hidden.size()
        vocab_size = logits.size(-1)
        hidden = hidden.view(batch_size*seq_len, model_dim)
        logits = logits.view(batch_size*seq_len, model_dim)

        # retrieval examples from database
        if self.training:
            distances, token_indices, searched_hidden = self.database.enhanced_search(hidden.cpu().numpy(),
                                                        top_k=self.top_k, retrieval_dropout=True)
        else:
            distances, token_indices, searched_hidden = self.database.enhanced_search(hidden.cpu().numpy(),
                                                        top_k=self.top_k, retrieval_dropout=False)
        
        distances = torch.FloatTensor(distances).to(hidden.device)
        token_indices = torch.LongTensor(token_indices).to(hidden.device)
        searched_hidden = torch.FloatTensor(searched_hidden).to(hidden.device)

        # compute dynamic database bandwidth 
        bandwidth = self.compute_bandwidth(hidden, searched_hidden)

        model_based_distribution = F.softmax(logits, dim=-1)
        vocab_size = model_based_distribution.size(-1)
        example_based_distribution, sparse_example_based_distribution = self.kernel.compute_example_based_distribution(distance, bandwidth, token_indices, vocab_size)

        mixing_weight = self.compute_mixing_weight(hidden, searched_hidden, sparse_example_based_distribution)

        # compute prediciton distribution by interpolating between model distribution and database distribution
        mixed_distribution = (1-mixing_weight)*model_based_distribution + mixing_weight*example_based_distribution
        log_probs = torch.log(mixing_weight)

        log_probs = log_probs.view(batch_size, seq_len, vocab_size).contiguous()
        return log_probs

        

