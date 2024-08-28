from typing import List, Tuple

import main
import filehandler
import utils


# pre-read data and save it
def pre_read(path: str) -> Tuple[List[List[int]], List[List[int]], List[int]]:
    x1_orig, x2_orig, y_orig = filehandler.read_data(path)
    x1_sentences_str = utils.split_sentences(x1_orig)
    x1_sentences_int = main.VOCAB.convert2indices(x1_sentences_str)
    x2_sentences_str = utils.split_sentences(x2_orig)
    x2_sentences_int = main.VOCAB.convert2indices(x2_sentences_str)
    return x1_sentences_int, x2_sentences_int, y_orig


if __name__ == '__main__':
    snli_path = 'data/snli_1.0'
    result = pre_read(f'{snli_path}/snli_1.0_dev.jsonl')
    filehandler.save_pickle(f'{snli_path}/snli_1.0_dev.pkl', result)
    result = pre_read(f'{snli_path}/snli_1.0_test.jsonl')
    filehandler.save_pickle(f'{snli_path}/snli_1.0_test.pkl', result)
    result = pre_read(f'{snli_path}/snli_1.0_train.jsonl')
    filehandler.save_pickle(f'{snli_path}/snli_1.0_train.pkl', result)
