import torch 
from torch import Tensor  

bleu_order = {}
for i in range(4):
    bleu_order["bleu-{}".format(i+1)] = 1
print(bleu_order)