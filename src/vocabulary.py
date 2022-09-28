# coding: utf-8
"""
Vocabulary module
"""
import logging
from pathlib import Path
import numpy as np
from typing import Dict, List
import unicodedata
from collections import Counter
from src.helps import flatten, read_list_from_file, sort_and_cut
from src.helps import write_list_to_file
import time
from src.constants import (
    UNK_TOKEN,
    UNK_ID,
    PAD_TOKEN,
    PAD_ID,
    BOS_TOKEN,
    BOS_ID,
    EOS_TOKEN,
    EOS_ID,
)

logger = logging.getLogger(__name__)

def build_vocab(data_cfg:Dict, datasets:List):
    """
    Build vocabulary for src side and trg side.
    Note: vocabulary either from file or dataset.
    """
    src_vocab = build_language_vocab(data_cfg["src"], datasets, data_cfg["src"]["language"])
    trg_vocab = build_language_vocab(data_cfg["trg"], datasets, data_cfg["trg"]["language"])

    return src_vocab, trg_vocab

def build_language_vocab(cfg, datasets, language):
    min_freq = cfg.get("vocab_min_freq", 1)
    max_size = cfg.get("vocab_max_size", -1)
    assert max_size > 0 and min_freq > 0

    vocab_file = cfg.get("vocab_file", None)
    if vocab_file is not None:
        unique_tokens = read_list_from_file(Path(vocab_file))
    elif datasets is not None:
        sentences = []
        for dataset in datasets:
            sentences.extend(dataset.tokernized_data[language])
        counter = Counter(flatten(sentences))
        # flatten(senteces) = list of list of tokens (nested)
        unique_tokens = sort_and_cut(counter, max_size, min_freq)
    else:
        raise Exception("Please provide dataset or vocab file to build a vocabulary.")
    
    start = time.time()
    vocab = Vocabulary(unique_tokens)
    end = time.time()
    logger.info("Spend time on get {} vocabulary = {}s".format(language, round(end-start,2)))
    assert len(vocab) <= max_size + len(vocab.specials)

    # check for all except for UNK token whether they are OOVs
    for s in vocab.specials[1:]:
        assert not vocab.is_unk(s)

    return vocab

class Vocabulary(object):
    """
    Vocabulary class mapping between tokens and indices.
    """
    def __init__(self, tokens: List[str], has_bos_eos: bool=True) -> None:
        "Create  vocabulary from list of tokens. :param tokens: list of tokens"
        if has_bos_eos:
            self.specials = [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]
        else:
            self.specials = [UNK_TOKEN, PAD_TOKEN]

        self._stoi: Dict[str, int] = {} # string to index
        self._itos: List[str] = []      # index to string

        # construct vocabulary
        self.add_tokens(tokens=self.specials + tokens)
        assert len(self._stoi) == len(self._itos)

        # assign special after stoi and itos are built
        self.pad_index = self.lookup(PAD_TOKEN)
        self.unk_index = self.lookup(UNK_TOKEN)
        assert self.pad_index == PAD_ID
        assert self.unk_index == UNK_ID
        if has_bos_eos:
            self.bos_index = self.lookup(BOS_TOKEN)
            self.eos_index = self.lookup(EOS_TOKEN)
            assert self.bos_index == BOS_ID
            assert self.eos_index == EOS_ID
        assert self._itos[UNK_ID] == UNK_TOKEN
    
    def lookup(self, token: str) -> int:
        "look up the encoding dictionary"
        return self._stoi.get(token, UNK_ID) 
    
    def add_tokens(self, tokens:List[str]) -> None:
        for token in tokens:
            token = self.normalize(token)
            new_index = len(self._itos)
            # add to vocabulary if not already there
            if token not in self._itos:
                self._itos.append(token)
                self._stoi[token] = new_index
    
    def is_unk(self,token:str) -> bool:
        """
        Check whether a token is covered by the vocabulary.
        """
        return self.lookup(token) == UNK_ID
    
    def to_file(self, file_path: Path) -> None:
        write_list_to_file(file_path, self._itos)
    
    def __len__(self) -> int:
        return len(self._itos)
    
    @staticmethod
    def normalize(token) -> str:
        return unicodedata.normalize('NFD', token)
    
    def array_to_sentence(self, array: np.ndarray, cut_at_eos: bool=True, skip_pad: bool=True) -> List[str]:
        """
        Convert an array of IDs to a sentences (list of tokens).
        array: 1D array containing indices
        Note: when cut_at_eos=True, sentence final token is </s>.
        """
        sentence = []
        for i in array:
            token = self._itos[i]
            if skip_pad and token == PAD_TOKEN:
                continue
            sentence.append(token)
            if cut_at_eos and token == EOS_TOKEN:
                break
        return sentence

    def arrays_to_sentences(self, arrays: np.ndarray, cut_at_eos: bool=True, skip_pad: bool=True) -> List[List[str]]:
        """
        Convert multiple arrays containing sequences of token IDs to their sentences.
        arrays: 2D array containing indices.
        return: list of list of tokens.
        """
        return [self.array_to_sentence(array=array, cut_at_eos=cut_at_eos, skip_pad=skip_pad) for array in arrays]

    def sentencens_to_ids(self, sentences:List[List[str]], bos:bool=False, eos:bool=False):
        """
        Return sentences_ids List[List[id]].
        """
        sentences_ids = []
        for sentence in sentences:
            sentence_ids = [self.lookup(token) for token in sentence]
            if bos is True:
                sentence_ids = [self.bos_index] + sentence_ids
            if eos is True:
                sentence_ids = sentence_ids + [self.eos_index]
            sentences_ids.append(sentence_ids)

        return sentences_ids

    def log_vocab(self, number:int) -> str:
        "First how many number of tokens in Vocabulary."
        return " ".join(f"({id}) {token}" for id, token in enumerate(self._itos[:number]))

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(len={self.__len__()}, "
                f"specials={self.specials})")
