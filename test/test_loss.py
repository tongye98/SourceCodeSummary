import torch 
import torch.nn.functional as F 
import torch.nn as nn 

x = torch.randn(2,4,5)
print(x)
target = torch.tensor([[0,2,3,4],[2,4,0,2]])
print(target)
one_hot = F.one_hot(target).float()
print(one_hot)
# softmax = torch.exp(x) / torch.sum(torch.exp(x),dim=-1)
softmax = torch.exp(x) / torch.sum(torch.exp(x),dim=-1).reshape(2,4,1)
print(softmax)
logsoftmax = torch.log(softmax)
print(logsoftmax)
print(one_hot*logsoftmax)
nllloss1 = -torch.sum(one_hot*logsoftmax, dim=-1)
print(nllloss1)
nllloss2 = -torch.sum(one_hot*logsoftmax)
print(nllloss2)
print('-'* 50)

# ###
logsoftmax = F.log_softmax(x, dim=-1)
loss = nn.NLLLoss(reduction='sum')
# [batch_size, vocab_size, seq_len]
nllloss = loss(logsoftmax.permute(0,2,1), target) # don't need to one-hot for target.
print(nllloss)
print('-'* 50)


### 
loss = nn.CrossEntropyLoss(reduction="sum")
crossentropy = loss(x.permute(0,2,1), target)
print(crossentropy)
print('-'* 50)
