from typing import Iterator, List, Tuple
import nltk


# split sentences to List[str], then combine them to List[List[str]] and return
def split_sentences(sentences: Iterator[str]) -> List[List[str]]:
    result = []
    for sentence in sentences:
        s = nltk.word_tokenize(sentence.lower())
        result.append(s)
    return result


# pad list of sentences according to the longest sentence in the batch
def pad_sentences(sentences: List[List[int]], pad_index: int, max_sentence_length: int) -> Tuple[List[List[int]], List[int]]:
    sentences_result = []
    length_result = []

    for sentence in sentences:
        length = len(sentence)
        # split the long sentence
        if length > max_sentence_length:
            s = sentence[0:max_sentence_length]
            length_result.append(max_sentence_length)
        else:
            s = [pad_index] * max_sentence_length
            s[0:length] = sentence
            length_result.append(length)
        sentences_result.append(s)

    return sentences_result, length_result
