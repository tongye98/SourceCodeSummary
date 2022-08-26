from torch.utils.data import Dataset, Sampler, DataLoader
from torch.utils.data import SequentialSampler, RandomSampler
import torch 

def test_SequentialSampler(dataset):
    print("-"*20+"Test SequentialSampler"+"-"*20)
    test_result = True
    sampler = SequentialSampler(data_source=dataset)
    for i, index in enumerate(sampler):
        print("index: {}, data: {}".format(str(index), str(dataset[index])))
        if i != index:
            test_result = False
    print("test SequentialSampler is {}".format("OK!" if test_result==True else "ERROR!"))

def test_RandomSampler(dataset, seed=980820):
    print("-"*20+"Test RandomSampler"+"-"*20)
    test_result = True 
    generator = torch.Generator()
    generator.manual_seed(seed)
    # generator to make sure every random sampler return same.
    sampler = RandomSampler(data_source=dataset, replacement=False, generator=generator)
    for i, index in enumerate(sampler):
        print("index: {}, data: {}".format(str(index), str(dataset[index])))
    print("test RamdomSampler is {}".format("OK!" if test_result == True else "ERROR!"))



if __name__ == "__main__":
    dataset = [17, 22, 3, 41, 8]
    test_SequentialSampler(dataset)
    test_RandomSampler(dataset)
