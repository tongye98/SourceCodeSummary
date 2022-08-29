import torch 
from torch import Tensor  


x = torch.randint(1,4,size=(2,3))
# print(x)

y = torch.tensor([[1,2,3],[11,12,14]])
print(y)
print(y.div(10, rounding_mode='floor'))
print(y.fmod(10))