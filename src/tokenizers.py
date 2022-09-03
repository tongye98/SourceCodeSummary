import logging
from typing import Dict, List
from src.helps import ConfigurationError
from pathlib import Path
logger = logging.getLogger(__name__)

from src.constants import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN

class BasicTokenizer(object):
    SPACE = chr(32)  # ' ': half-width white space (ascii)
    SPACE_ESCAPE = chr(9601)  # '▁': sentencepiece default
    SPECIALS = [BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN]

    def __init__(self, level:str="word", lowercase:bool=False,
                 normalize: bool=False, max_length: int=-1, min_length:int=1,
                 filter_or_truncate: str="truncate") -> None:
        self.level = level
        self.lowercase = lowercase
        self.normalize = normalize
        self.max_length = max_length
        self.min_length = min_length
        self.filter_or_truncate = filter_or_truncate
        assert self.filter_or_truncate in ["filter", "truncate"], "Invalid filter_or_truncate!"

    
    def pre_process(self, sentence:str) -> str:
        """
        Pre-process setence. 
        Lowercase, normalize.
        """
        if self.normalize:
            sentence = sentence.strip()
        if self.lowercase:
            sentence = sentence.lower()
        
        return sentence
    
    def post_process(self, sentence: List[str], generate_unk: bool=True) -> str:
        """
        Post-process sentence tokens.
        result a sentence(a string.)
        """
        sentence = self.remove_special(sentence, generate_unk=generate_unk)
        sentence = self.SPACE.join(sentence)

        return sentence

    def remove_special(self, sentence: List[str], generate_unk: bool=False) -> List[str]:
        specials = self.SPECIALS[:-1] if generate_unk else self.SPECIALS
        return [token for token in sentence if token not in specials]

    def filter_or_truncate_by_length(self, sentence_token:List[str]) -> List[str]:
        if self.filter_or_truncate == "filter":
            if len(sentence_token) < self.min_length or len(sentence_token) > self.max_length:
                sentence_token = None
        elif self.filter_or_truncate == "truncate":
            sentence_token = sentence_token[:self.max_length]
        else:
            return None 

        return sentence_token

    def __call__(self, sentence: str) -> List[str]:
        """
        Tokenize single sentence.
        """
        sentence_token = sentence.split(self.SPACE)
        sentence_token = self.filter_or_truncate_by_length(sentence_token)

        return sentence_token

    def __repr__(self):
        return (f"{self.__class__.__name__}(level={self.level}, "
                f"lowercase={self.lowercase}, normalize={self.normalize}, "
                f"filter_by_length=({self.min_length}, {self.max_length}))")

import sentencepiece as sp
class SentencePieceTokenizer(BasicTokenizer):
    def __init__(self, level: str = "bpe", lowercase: bool = False, normalize: bool = False, 
                 max_length: int = -1, min_length: int = 1) -> None:
        super().__init__(level, lowercase, normalize, max_length, min_length)
        assert self.level == "bpe"
        self.model_file = Path(None)
        assert self.model_file.is_file(), "spm model file not found."
        self.spm = sp.SentencePieceProcessor()
        self.spm.load(self.model_file)
    


class SubwordNMTTokenizer(BasicTokenizer):
    def __init__(self, level: str = "word", lowercase: bool = False, normalize: bool = False, 
                 max_length: int = -1, min_length: int = 1) -> None:
        super().__init__(level, lowercase, normalize, max_length, min_length)


def build_tokenizer(data_cfg: Dict) -> Dict[str,BasicTokenizer]:
    src_language = data_cfg["src"]["language"]
    trg_language = data_cfg["trg"]["language"]
    tokenizer = {
        src_language: build_language_tokenizer(data_cfg["src"]),
        trg_language: build_language_tokenizer(data_cfg["trg"]),
    }
    logger.info("%s tokenizer: %s", src_language, tokenizer[src_language])
    logger.info("%s tokenizer: %s", trg_language, tokenizer[trg_language])
    return tokenizer

def build_language_tokenizer(cfg: Dict):
    """
    Build tokenizer.
    """
    tokenizer = None 

    if cfg["level"] == "word":
        tokenizer = BasicTokenizer(level=cfg["level"],lowercase=cfg["lowercase"],
                                   normalize=cfg["normalize"], max_length=cfg["max_length"],
                                   min_length=cfg["min_length"], filter_or_truncate=cfg["filter_or_truncate"])
    elif cfg["level"] == "bpe":
        tokenizer_type = cfg.get("tokenizer_type", "sentencepiece")
        if tokenizer_type == "sentencepiece":
            tokenizer = SentencePieceTokenizer()
        elif tokenizer_type == "subword-nmt":
            tokenizer  = SubwordNMTTokenizer()
        else:
            raise ConfigurationError("Unkonwn tokenizer type.")
    else:
        raise ConfigurationError("Unknown tokenizer level.")
    
    return tokenizer
