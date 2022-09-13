from typing import Callable, Dict, List
from src.constants import EOS_TOKEN
from src.helps import ConfigurationError, read_list_from_file
from torch.utils.data import Dataset
from pathlib import Path
from src.vocabulary import Vocabulary

def build_dataset(dataset_type: str, path:str, split_mode:str,
                  src_language: str, trg_language: str, tokenizer: Dict) -> Dataset:
    """
    Build a dataset.
    """
    dataset = None 
    if dataset_type == "plain":
        dataset = PlaintextDataset(path, split_mode, src_language, trg_language, tokenizer)
    elif dataset_type == "other":
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
        raise NotImplementedError
    
    def get_item(self, idx:int, language:str):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(len={self.__len__()}, "
                f"src_lang={self.src_language}, trg_lang={self.trg_language})")

class PlaintextDataset(BaseDataset):
    def __init__(self, path: str, split_mode: str, src_language: str, trg_language: str, 
                 tokenizer: Dict) -> None:
        super().__init__(path, split_mode, src_language, trg_language)

        self.tokenizer = tokenizer

        self.original_data = self.load_data(path)
        self.tokernized_data = self.tokenize_data(self.original_data)
        self.src_vocabs, self.src_maps, self.alignments = self.get_src_vocabs_source_maps_alignments(self.tokernized_data)
        self.tokernized_data_ids = None 
    
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
    
    def tokenize_data(self, original_data: Dict[str, List]):
        """
        Tokenize data
        tokenize_data["en"] = [["hello", "word"], ["x", "x"]]
        """
        tokenize_data = dict()
        tokenize_data[self.src_language] = [self.tokenizer[self.src_language](sentence) for sentence in original_data[self.src_language]]
        tokenize_data[self.trg_language] = [self.tokenizer[self.trg_language](sentence) for sentence in original_data[self.trg_language]]
        return tokenize_data

    def tokernized_data_to_ids(self, src_vocab:Vocabulary, trg_vocab:Vocabulary) -> None:
        """
        self.tokernized_data_ids: dict
        self.tokernized_data_ida["en"] = [[3,4], [5,6]]
        """
        self.tokernized_data_ids = dict()
        self.tokernized_data_ids[self.src_language] = src_vocab.sentencens_to_ids(self.tokernized_data[self.src_language], bos=False, eos=True)
        self.tokernized_data_ids[self.trg_language] = trg_vocab.sentencens_to_ids(self.tokernized_data[self.trg_language], bos=True, eos=True)

        
    def get_src_vocabs_source_maps_alignments(self, tokernized_data):
        src_vocabs = list()
        src_maps = list()
        alignments = list()

        src_tokernized_data = tokernized_data[self.src_language]
        for i, sentence_tokens in enumerate(src_tokernized_data):
            src_vocab = Vocabulary(tokens=sentence_tokens, has_bos_eos=False)
            src_vocabs.append(src_vocab)
            src_map = [src_vocab.lookup(token) for token in sentence_tokens + [EOS_TOKEN]] # no bos, has eos
            src_maps.append(src_map)
            alignment = [src_vocab.lookup(token) for token in tokernized_data[self.trg_language][i] + [EOS_TOKEN]]
            alignments.append(alignment)

        return src_vocabs, src_maps, alignments

    def __getitem__(self, index):
        """
        src: [id, id, id, ...]
        trg: [id, id, id, ...]
        return (src, trg) tuple
        """
        src = self.tokernized_data_ids[self.src_language][index]
        trg = self.tokernized_data_ids[self.trg_language][index]
        copy_param = dict()
        copy_param["src_vocab"] = self.src_vocabs[index]
        copy_param["src_map"] = self.src_maps[index]
        copy_param["alignment"] = self.alignments[index]
        return (src, trg, copy_param)
    
    def __len__(self) -> int:
        return len(self.original_data[self.src_language])
    
    @property
    def src(self) -> List[str]:
        sentence_list = []
        for i in range(self.__len__()):
            sentence = self.original_data[self.src_language][i]
            sentence_list.append(sentence)
        return sentence_list
    
    @property
    def trg(self) -> List[str]:
        sentence_list = []
        for i in range(self.__len__()):
            sentence = self.original_data[self.trg_language][i]
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
