from helps import ConfigurationError
from torch.utils.data import Dataset
from pathlib import Path
def build_dataset(dataset_type: str="other", path:str,
                  src_language: str, trg_language: str):
    """
    Build a dataset.
    """
    dataset = None 
    if dataset_type == "plain":
        dataset = PlaintextDataset()
    elif dataset_type == "other":
        # TODO need to expand
        raise NotImplementedError
    else:
        raise ConfigurationError("Invalid dataset_type.")

    return dataset

class BaseDataset(Dataset):
    """
    BaseDataset which loads and looks up data.
    """
    def __init__(self, path:str,
                 src_language:str, trg_language:str) -> None:
        super().__init__()
        self.path = path,
        self.src_language = src_language
        self.trg_language = trg_language


class PlaintextDataset(BaseDataset):
    def __init__(self, path: str, src_language: str, trg_language: str) -> None:
        super().__init__(path, src_language, trg_language)

        self.load_data(path)
    
    def load_data(self,path:str):
        path = Path(path)
