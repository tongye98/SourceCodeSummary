from helps import ConfigurationError, read_list_from_file
from torch.utils.data import Dataset
from pathlib import Path
def build_dataset(dataset_type: str, path:str,
                  src_language: str, trg_language: str):
    """
    Build a dataset.
    """
    dataset = None 
    if dataset_type == "plain":
        dataset = PlaintextDataset(path, src_language, trg_language)
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

    def __len__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(len={self.__len__()}, "
                f"src_lang={self.src_language}, trg_lang={self.trg_language})")

class PlaintextDataset(BaseDataset):
    def __init__(self, path: str, src_language: str, trg_language: str) -> None:
        super().__init__(path, src_language, trg_language)

        self.data = self.load_data(path)
    
    def load_data(self,path:str):
        """"
        loda data and tokenize data.
        return data(is a dict)
            data["en"] = ["hello world","xxx"]
            data["de"] = ["i am", "xxx"]
        """
        def pre_process(sentences_list, language):
            if self.tokenizer[language] is not None:
                sentences_list = [self.tokenizer[language].pre_process(sentence) 
                                    for sentence in sentences_list if len(sentence) > 0]

        path = Path(path)

        src_file = path.with_suffix(f"{path.suffix}.{self.src_language}")
        assert src_file.is_file(), "src file not found."
        src_list = read_list_from_file(src_file)
        data = {self.src_language: pre_process(src_list, self.src_language)}

        trg_file = path.with_suffix(f"{path.suffix}.{self.trg_language}")
        assert trg_file.is_file(), "trg file not found."
        trg_list = read_list_from_file(trg_file)
        data[self.trg_language] = pre_process(trg_list, self.trg_language)

        assert len(data[self.src_language]) == len(data[self.trg_language])

        return data
    
    def get_item(self, idx:int, language:str) -> str:
        return self.data[language][idx]
    
    def __len__(self) -> int:
        return len(self.data[self.src_language])


if __name__ == "__main__":
    build_dataset("plain","data/train",'code','summary')
    # dataset
    #       |- train.code
    #       |- train.summary
    #       |- dev.code
    #       |- dev.summary
    #       |- test.code
    #       |- test.summary
    # train_data_path: dataset/train
    # dev_data_path: dataset/dev
    # test_data_path: dataset/test
