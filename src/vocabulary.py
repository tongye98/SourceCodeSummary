# coding: utf-8
"""
Vocabulary module
"""
import logging
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple
import unicodedata
import sys
from collections import Counter
from src.helps import flatten, sort_and_cut
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

def build_vocab(data_cfg:Dict, dataset):
    """
    Build vocabulary for src side and trg side.
    Note: vocabulary either from file(todo) or dataset.
    """

    src_vocab = build_language_vocab(data_cfg["src"], dataset, data_cfg["src"]["language"])
    trg_vocab = build_language_vocab(data_cfg["trg"], dataset, data_cfg["trg"]["language"])

    return src_vocab, trg_vocab

def build_language_vocab(cfg, dataset, language):

    min_freq = cfg.get("vocab_min_freq", 1)
    max_size = cfg.get("vocab_max_size", sys.maxsize)
    assert max_size > 0

    if dataset is not None:
        # FIXME how to use tokenizer.
        sentences = dataset.tokernized_data[language]
        # senteces: list of list of tokens (nested)
        counter = Counter(flatten(sentences))
        unique_tokens = sort_and_cut(counter, max_size, min_freq)
    else:
        raise Exception("Please provide dataset to build a vocabulary.")
    
    start = time.time()
    vocab = Vocabulary(unique_tokens)
    end = time.time()
    logger.info("Spend time on get vocabulary = {}s".format(round(end-start,2)))
    assert len(vocab) <= max_size + len(vocab.specials)

    # check for all except for UNK token whether they are OOVs
    for s in vocab.specials[1:]:
        assert not vocab.is_unk(s)

    return vocab

class Vocabulary(object):
    """vocabulary class mapping between tokens and indices."""
    def __init__(self, tokens: List[str], has_bos_eos: bool=True) -> None:
        "Create  vocabulary from list of tokens. :param tokens: list of tokens"
        if has_bos_eos:
            self.specials = [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]
        else:
            self.specials = [UNK_TOKEN, PAD_TOKEN]

        self._stoi: Dict[str, int] = {} # string to index
        self._itos: List[str] = [] # index to string

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
            # token = self.normalize(token)
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
        return unicodedata.normalize('NFD',token)
    
    def array_to_sentence(self, 
                          array: np.ndarray, 
                          cut_at_eos: bool=True,
                          skip_pad: bool=True) -> List[str]:
        """
        convert an array of IDs to a sentences(list of strings/tokens).
        array: 1D array containing indices
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

    def arrays_to_sentences(self,
                            arrays: np.ndarray,
                            cut_at_eos: bool=True,
                            skip_pad: bool=True) -> List[List[str]]:
        """
        convert multiple arrays containing sequences of token IDs to their sentences.
        arrays: 2D array containing indices
        return: list of list of strings(tokens)
        """
        return [self.array_to_sentence(array=array, cut_at_eos=cut_at_eos, skip_pad=skip_pad) 
                for array in arrays]
    
    # def sentences_to_ids(self, sentences:List[List[str]], bos:bool=False, eos:bool=False) -> Tuple[List[List[int]], List[int]]:
    #     """
    #     Encode sentences to indices and pad sentences to the maximum length 
    #     of the sentences given.
    #     Used in collate_fn function.
    #     return 
    #         - padded ids
    #         - original lengths before padding(but include bos and eos token)
    #     """
    #     max_len = max([len(sentence) for sentence in sentences])
    #     if bos is True:
    #         max_len += 1
    #     if eos is True:
    #         max_len += 1
        
    #     padded_sentences = []
    #     sentences_lengths = []
    #     for sentence in sentences:
    #         sentence_ids = [self.lookup(token) for token in sentence]
    #         if bos is True:
    #             sentence_ids = [self.bos_index] + sentence_ids
    #         if eos is True:
    #             sentence_ids = sentence_ids + [self.eos_index]
    #         pad_number = max_len - len(sentence_ids)
    #         assert pad_number >= 0, "pad number < 0, must Error!"
    #         padded_sentences.append(sentence_ids + [self.pad_index] * pad_number)
    #         sentences_lengths.append(len(sentence_ids)) # sentence_length include bos and eos token, but not include pad token.
        
    #     return padded_sentences, sentences_lengths

    def sentencens_to_ids(self, sentences:List[List[str]]):
        """
        return sentences_ids List[List[id]]
        """
        sentences_ids = []
        for sentence in sentences:
            sentence_ids = [self.lookup(token) for token in sentence]
            sentences_ids.append(sentence_ids)

        return sentences_ids

    def log_vocab(self, number:int) -> str:
        "First number tokens in Vocabulary"
        return " ".join(f"({id}) {token}" for id, token in enumerate(self._itos[:number]))

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(len={self.__len__()}, "
                f"specials={self.specials})")

if __name__ == "__main__":
    vocab = Vocabulary(tokens=['hello','world'])
    print(vocab)
    array = np.array([4,5])
    print(vocab.array_to_sentence(array))
