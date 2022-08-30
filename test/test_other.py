import torch 
from torch import Tensor  


x = torch.randint(1,4,size=(2,3))
# print(x)

y = torch.tensor([[1,2,3],[11,12,14]])
print(y)
print(y.div(10, rounding_mode='floor'))
print(y.fmod(10))


# 创建2D张量
b = torch.arange(0, 9).view([3, 3])
print(b)
# 获取2D张量的第2个维度且索引号为0和1的张量子集(第一列和第二列)
print(torch.index_select(b, dim = 0, index = torch.tensor([0, 1, 1])))



print(torch.nonzero(torch.tensor([1, 1, 1, 0, 1]), as_tuple=False).view(-1))


v = torch.tensor([1, 1, 1, 0, 1])
print(v[1].item())