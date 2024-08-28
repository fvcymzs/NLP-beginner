from typing import List, Optional, Dict, Tuple
import torch

import utils


# convert word to int
class Vocab:
    # create a dictionary word2id[word] = index
    def __init__(self, word2id: Optional[Dict[str, int]] = None):
        if word2id is not None:
            self.word2id = word2id
        else:
            self.word2id = dict()
            self.add('<pad>')  # pad token
            self.add('<unk>')  # unknown token

        self.pad_index = self.word2id['<pad>']
        self.unk_index = self.word2id['<unk>']

    # compute number of words in VocabEntry.
    def __len__(self) -> int:
        return len(self.word2id)

    # add word to VocabEntry, if it is previously unseen
    def add(self, word: str) -> int:
        if word not in self.word2id:
            index = len(self.word2id)
            self.word2id[word] = index
            return index
        else:
            return self.word2id[word]

    # retrieve word's index, return the index for the unk token if the word is out of vocabulary
    def get(self, word: str) -> int:
        return self.word2id.get(word, self.unk_index)

    # convert list of sentences of words into list of list of indices
    def words2indices(self, sentences: List[List[str]]) -> List[List[int]]:
        return [[self.get(word) for word in sentence] for sentence in sentences]

    # convert list of sentences (words) into tensor with necessary padding for shorter sentences
    def to_input_tensor(self, sentences: List[List[str]], max_sentence_length: int, device: Optional[torch.device]) -> Tuple[torch.Tensor, List[int]]:
        word_ids = self.words2indices(sentences)
        list_sentences, seq_lengths = utils.pad_sentences(word_ids, self.pad_index, max_sentence_length)
        sentences_var = torch.tensor(list_sentences, dtype=torch.long, device=device)
        return sentences_var, seq_lengths

    # convert list of sentences (words) into list
    def convert2indices(self, sentences: List[List[str]]) -> List[List[int]]:
        word_ids = self.words2indices(sentences)
        return word_ids
