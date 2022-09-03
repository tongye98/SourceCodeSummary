from typing import Callable, Dict, List
from src.helps import ConfigurationError, read_list_from_file
from torch.utils.data import Dataset
from pathlib import Path

def build_dataset(dataset_type: str, path:str, split_mode:str,
                  src_language: str, trg_language: str,
                  tokenizer: Dict, sentences_to_vocab_ids:Dict[str, Callable] = None) -> Dataset:
    """
    Build a dataset.
    """
    dataset = None 
    if dataset_type == "plain":
        dataset = PlaintextDataset(path, split_mode, src_language, trg_language, tokenizer, sentences_to_vocab_ids)
    elif dataset_type == "other":
        # TODO need to expand
        raise NotImplementedError
    else:
        raise ConfigurationError("Invalid dataset_type.")

    return dataset


# BaseDataset is child of torch.utils.data.Dataset.
class BaseDataset(Dataset):
    """
    BaseDataset which loads and looks up data.
    """
    def __init__(self, path:str, split_mode: str,
                 src_language:str, trg_language:str) -> None:
        super().__init__()
        self.path = path
        self.split_mode = split_mode
        self.src_language = src_language
        self.trg_language = trg_language

    def __getitem__(self, index):
        src = self.get_item(idx=index, language=self.src_language)
        trg = self.get_item(idx=index, language=self.trg_language)
        return (src, trg)
    
    def get_item(self, idx:int, language:str):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(len={self.__len__()}, "
                f"src_lang={self.src_language}, trg_lang={self.trg_language})")

class PlaintextDataset(BaseDataset):
    def __init__(self, path: str, split_mode: str, src_language: str, trg_language: str, 
                 tokenizer: Dict, sentences_to_vocab_ids: Dict[str, Callable] = None) -> None:
        super().__init__(path, split_mode, src_language, trg_language)

        self.tokenizer = tokenizer
        # place holder for senteces_to_vocab_ids
        place_holder = {self.src_language:None, self.trg_language:None}
        self.sentences_to_vocab_ids = place_holder if sentences_to_vocab_ids is None else sentences_to_vocab_ids

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
            return sentences_list

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
    
    def get_item(self, idx:int, language:str) -> List[str]:
        # call tokernizer to process the sampled sample.
        return self.tokenizer[language](self.data[language][idx])
    
    def __len__(self) -> int:
        return len(self.data[self.src_language])
    
    @property
    def src(self) -> List[str]:
        sentence_list = []
        for i in range(self.__len__()):
            sentence = self.data[self.src_language][i]
            sentence_list.append(sentence)
        return sentence_list
    
    @property
    def trg(self) -> List[str]:
        sentence_list = []
        for i in range(self.__len__()):
            sentence = self.data[self.trg_language][i]
            sentence_list.append(sentence)
        return sentence_list
        

if __name__ == "__main__":
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

    path = "data/rencos_python/train"
    train_data = build_dataset(dataset_type="plain", path=path, src_language="code",trg_language="summary")
    print(train_data)
    print(len(train_data))
    print(train_data[10])
