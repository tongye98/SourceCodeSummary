from typing import Tuple
import torch 
import torch.nn as nn
from src.database import Database, EnhancedDatabase 
import math
import torch.nn.functional as F
from src.kernel import Kernel, GaussianKernel, LaplacianKernel

class Retriever(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, hidden: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """
        hidden: [batch_size, trg_len, model_dim]
        logits: [batch_size, trg_len, vocab_size]
        return:
            log_probs: [batch_size, seq_len, vocab_size]
        """
        raise NotImplementedError("The forward method is not implemented in the Retrieval class.")
    
    def detailed_forward(self, hidden:torch.Tensor, logits:torch.Tensor) -> Tuple[torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        hidden: [batch_size, trg_len, model_dim]
        logits: [batch_size, trg_len, vocab_size]
        """
        raise NotImplementedError

class NoRetriever(Retriever):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, hidden: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs

class StaticRetriever(Retriever):
    def __init__(self, database:Database, top_k:int, mixing_weight:float, kernel:Kernel, bandwidth:float) -> None:
        super().__init__()
        self.database = database
        self.top_k = top_k 
        self.mixing_weight = mixing_weight
        self.kernel = kernel
        self.bandwidth = bandwidth
    
    def forward(self, hidden:torch.Tensor, logits:torch.Tensor) -> torch.Tensor:
        """
        hidden [batch_size, trg_len, model_dim]
        logits [batch_size, trg_len, trg_vocab_size]
        """
        batch_size, trg_len, model_dim = hidden.size()
        vocab_size = logits.size(-1)
        hidden = hidden.view(batch_size*trg_len, model_dim)
        logits = logits.view(batch_size*trg_len, vocab_size)

        model_based_distribution = F.softmax(logits, dim=-1)
        # model_based_distribution [batch_size*trg_len, trg_vocab_size]
        distances, token_indices = self.database.search(hidden.cpu().numpy(), top_k=self.top_k)
        # distances [batch_size*trg_len, top_k] distance
        # token_indices [batch_size*trg_len, top_k] id
        distances = torch.FloatTensor(distances).to(hidden.device)
        token_indices = torch.LongTensor(token_indices).to(hidden.device)
        example_based_distribution, _ = self.kernel.compute_example_based_distribution(distances, self.bandwidth, token_indices, vocab_size)
        # example_based_distribution [batch_size*trg_len, trg_vocab_size]
        mixed_distribution = (1 - self.mixing_weight)*model_based_distribution + self.mixing_weight*example_based_distribution
        final_distribution = mixed_distribution.view(batch_size, trg_len, vocab_size).contiguous()
        # final_distribution [batch_size, trg_len, vocab_size]
        return final_distribution

class DynamicRetriever(Retriever):
    def __init__(self, database:Database, top_k:int, kernel:Kernel) -> None:
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
        """
        hidden [batch_size, trg_len, model_dim]
        logits [batch_size, trg_len, trg_vocab_size]
        """
        batch_size, trg_len, model_dim = hidden.size()
        vocab_size = logits.size(-1)
        hidden = hidden.view(batch_size*trg_len, model_dim)
        logits = logits.view(batch_size*trg_len, vocab_size)

        # retrieval examples from database
        if self.training:
            distances, token_indices, searched_hidden = self.database.enhanced_search(hidden.cpu().numpy(),
                                                        top_k=self.top_k, retrieval_dropout=True)
        else:
            distances, token_indices, searched_hidden = self.database.enhanced_search(hidden.cpu().numpy(),
                                                        top_k=self.top_k, retrieval_dropout=False)
        
        distances = torch.FloatTensor(distances).to(hidden.device)
        # distances [batch_size*trg_len, top_k]
        token_indices = torch.LongTensor(token_indices).to(hidden.device)
        # token_indices [batch_size*trg_len, top_k]
        searched_hidden = torch.FloatTensor(searched_hidden).to(hidden.device)
        # searched_hidden [batch_size*trg_len, top_k, model_dim]

        # compute dynamic database bandwidth 
        bandwidth = self.compute_bandwidth(hidden, searched_hidden)
        # bandwidth [batch_size*trg_len, 1]

        model_based_distribution = F.softmax(logits, dim=-1)
        # model_based_distribution [batch_size*trg_len, trg_vocab_size]
        vocab_size = model_based_distribution.size(-1)
        example_based_distribution, sparse_example_based_distribution = self.kernel.compute_example_based_distribution(distances, bandwidth, token_indices, vocab_size)
        # example_based_distribution [batch_size*trg_len, trg_vocab_size]
        # sparse_example_based_distribution [batch_size*trg_len, top_k] 
        mixing_weight = self.compute_mixing_weight(hidden, searched_hidden, sparse_example_based_distribution)
        # mixing_weight [batch_size*trg_len, 1]
        # compute prediciton distribution by interpolating between model distribution and database distribution
        mixed_distribution = (1-mixing_weight)*model_based_distribution + mixing_weight*example_based_distribution
        # mixed_distribution [batch_size*trg_len, trg_vocab_size]

        mixed_distribution = mixed_distribution.view(batch_size, trg_len, vocab_size).contiguous()
        return mixed_distribution
    
    def compute_bandwidth(self, hidden:torch.Tensor, searched_hidden:torch.Tensor) -> torch.Tensor:
        """
        hidden [batch_size*trg_len, model_dim]
        searched_hidden [batch_size*trg_len, top_k, model_dim]
        """
        mean_hidden = searched_hidden.mean(dim=1)
        # mean_hidden [batch_size*trg_len, model_dim]
        bandwidth = torch.exp(self.bandwidth_estimator(torch.cat([hidden, mean_hidden], dim=-1)))
        # bandwidth [batch_size*trg_len, 1]
        return bandwidth

    def compute_mixing_weight(self, hidden:torch.Tensor, searched_hidden:torch.Tensor, sparse_probs: torch.Tensor)->torch.Tensor:
        """
        hidden: [batch_size*trg_len, model_dim]
        searched_hidden [batch_size*trg_len, top_k, model_dim]
        sparse_probs: [batch_size*trg_len, top_k]
        return
            mixing_weight [batch_size*trg_len, 1]
        """
        merged_hidden = sparse_probs.unsqueeze(1).contiguous().matmul(searched_hidden).squeeze(1)
        # merged_hidden [batch_size*trg_len, model_dim]
        mixing_weight = torch.sigmoid(self.mixing_weight_estimator(torch.cat([hidden, merged_hidden], dim=-1)))
        return mixing_weight

    def detailed_forward(self, hidden: torch.Tensor, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        hidden [batch_size, seq_len, hidden_size]
        logits [batch_size, seq_len, hidden_size]
        return 
        """
        # reshape hidden and logits for knn retrieval
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
        example_based_distribution, sparse_example_based_distribution = self.kernel.compute_example_based_distribution(
                                                                            distances, token_indices, vocab_size)
        
        mixing_weight = self.compute_mixing_weight(hidden, searched_hidden, sparse_example_based_distribution)

        # compute prediction distribution by interpolating between model distribution and database distribution
        mixed_distribution = (1-mixing_weight)*model_based_distribution + mixing_weight*example_based_distribution

        mixed_distribution = mixed_distribution.view(batch_size, seq_len, vocab_size).contiguous()
        model_based_distribution = model_based_distribution.view(batch_size, seq_len, vocab_size).contiguous()
        example_based_distribution = example_based_distribution.view(batch_size, seq_len, vocab_size).contiguous()
        mixing_weight = mixing_weight.squeeze(-1).view(batch_size, seq_len).contiguous()
        bandwidth = bandwidth.squeeze(-1).view(batch_size, seq_len).contiguous()

        return mixed_distribution, model_based_distribution, example_based_distribution, mixing_weight, bandwidth


def build_retriever(cfg: dict) -> Retriever:
    retriever_cfg = cfg["retriever"]
    retriever_type = retriever_cfg["type"]

    if retriever_type == "no_retriever":
        retriever = NoRetriever()
    elif retriever_type == "static_retriever":
        database = Database(index_path=retriever_cfg["index_path"], token_map_path=retriever_cfg["token_map_path"])
        retriever = StaticRetriever(database=database, top_k=retriever_cfg["top_k"], mixing_weight=retriever_cfg["mixing_weight"],
                                    bandwidth=retriever_cfg["bandwidth"], 
                                    kernel=GaussianKernel() if retriever_cfg["kernel"] == "gaussian" else LaplacianKernel())
    elif retriever_type == "dynamic_retriever":
        database = EnhancedDatabase(index_path=retriever_cfg["index_path"], token_map_path=retriever_cfg["token_map_path"],
                                    embedding_path=retriever_cfg["emebedding_path"], in_memory=retriever_cfg["in_memory"])
        retriever = DynamicRetriever(database=database, top_k=retriever_cfg["top_k"],
                                    kernel=GaussianKernel() if retriever_cfg["kernel"] == "gaussian" else LaplacianKernel())
    else:
        raise ValueError("The {} is not supported currently.".format(retriever_type))
    
    return retriever