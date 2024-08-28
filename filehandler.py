from typing import List, Tuple, Dict, Any
import json
import pickle


# read data file
def read_data(path: str) -> Tuple[List[str], List[str], List[int]]:
    index = {'contradiction': 0, 'entailment': 1, 'neutral': 2}
    x1 = []
    x2 = []
    labels = []
    with open(path, 'r') as f:
        for line in f:
            line = json.loads(line)
            if line['gold_label'] == '-':
                # skip for unknown label
                continue
            x1.append(line['sentence1'])
            x2.append(line['sentence2'])
            labels.append(index[line['gold_label']])
    return x1, x2, labels


# read glove file
def read_glove(path: str) -> Tuple[List[List[float]], Dict[str, int]]:
    embedding = []
    word2id = {}
    with open(path, 'r', encoding='UTF-8') as f:
        for index, line in enumerate(f):
            line = line.rstrip('\n')
            # skip empty line
            if line:
                list_line = line.split()
                embedding.append([float(value) for value in list_line[1:]])
                word2id[list_line[0]] = index
    return embedding, word2id


# save pickle file
def save_pickle(path: str, data: Any) -> None:
    with open(path, 'wb') as f:
        pickle.dump(data, f)


# load pickle file
def load_pickle(path: str) -> Any:
    with open(path, 'rb') as f:
        return pickle.load(f)
