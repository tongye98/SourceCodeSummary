import torch 

src_mask = torch.randint(1,3,(2,1,4))
print(src_mask)

trg_mask = src_mask.new_ones((1,1,1))
print(trg_mask)