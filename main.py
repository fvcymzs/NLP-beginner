import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

import filehandler
from vocab import Vocab
from dataset import MyDataset
import esim


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDING_SIZE = 100
MAX_SENTENCE_LENGTH = 48
EPOCH = 5
BATCH_SIZE = 150
HIDDEN_SIZE = 200
CLASS_NUM = 3
LEARNING_RATE = 4e-4

print('loading pre-trained word vectors...')
list_embedding, word2id = filehandler.read_glove("data/glove.6B.100d.txt")
# generate pad and unk manually
pad_index = len(list_embedding)
list_embedding.append(torch.zeros(EMBEDDING_SIZE, dtype=torch.float).tolist())
word2id['<pad>'] = pad_index
unk_index = len(list_embedding)
list_embedding.append(torch.randn(EMBEDDING_SIZE).tolist())
word2id['<unk>'] = unk_index
VOCAB = Vocab(word2id)
tensor_embedding = torch.tensor(list_embedding).to(DEVICE)
assert tensor_embedding.shape[1] == EMBEDDING_SIZE
EMBEDDING = nn.Embedding.from_pretrained(tensor_embedding, freeze=False, padding_idx=VOCAB.pad_index)
print('finished')


def evaluate(model: esim.ESIM, dataloader: DataLoader, dataset: MyDataset) -> float:
    sum_num = len(dataset)
    corr_num = 0
    model.eval()
    with torch.no_grad():
        for (x1, seq_lengths1, x2, seq_lengths2, y) in tqdm(dataloader, total=len(dataloader)):
            x1 = x1.to(device=DEVICE)
            seq_lengths1 = seq_lengths1.to(device=DEVICE)
            x2 = x2.to(device=DEVICE)
            seq_lengths2 = seq_lengths2.to(device=DEVICE)
            y = y.to(device=DEVICE)
            scores = model.get_scores(x1, seq_lengths1, x2, seq_lengths2)
            corr_num += torch.eq(scores.argmax(dim=1), y).sum().item()
    cur_accuracy = corr_num / sum_num
    return cur_accuracy


def train():
    # read data
    snli_path = 'data/snli_1.0'
    x1_train_list_list_int, x2_train_list_list_int, y_train_orig = filehandler.load_pickle(f'{snli_path}/snli_1.0_train.pkl')
    x1_val_list_list_int, x2_val_list_list_int, y_val_orig = filehandler.load_pickle(f'{snli_path}/snli_1.0_dev.pkl')

    # handle data
    x1_train_tensor = torch.nn.utils.rnn.pad_sequence([torch.tensor(sentence, dtype=torch.long, device=None) for sentence in x1_train_list_list_int],
                                                      batch_first=True,
                                                      padding_value=VOCAB.pad_index)
    x1_train_seq_lengths = torch.tensor([len(sentence) for sentence in x1_train_list_list_int])
    x2_train_tensor = torch.nn.utils.rnn.pad_sequence([torch.tensor(sentence, dtype=torch.long, device=None) for sentence in x2_train_list_list_int],
                                                      batch_first=True,
                                                      padding_value=VOCAB.pad_index)
    x2_train_seq_lengths = torch.tensor([len(sentence) for sentence in x2_train_list_list_int])
    x1_val_tensor = torch.nn.utils.rnn.pad_sequence([torch.tensor(sentence, dtype=torch.long, device=None) for sentence in x1_val_list_list_int],
                                                    batch_first=True,
                                                    padding_value=VOCAB.pad_index)
    x1_val_seq_lengths = torch.tensor([len(sentence) for sentence in x1_val_list_list_int])
    x2_val_tensor = torch.nn.utils.rnn.pad_sequence([torch.tensor(sentence, dtype=torch.long, device=None) for sentence in x2_val_list_list_int],
                                                    batch_first=True,
                                                    padding_value=VOCAB.pad_index)
    x2_val_seq_lengths = torch.tensor([len(sentence) for sentence in x2_val_list_list_int])
    y_train = torch.tensor(y_train_orig)
    y_val = torch.tensor(y_val_orig)
    print(x1_train_tensor.shape, x2_train_tensor.shape, torch.max(x1_train_seq_lengths), torch.max(x2_train_seq_lengths))
    print(x1_val_tensor.shape, x2_val_tensor.shape, torch.max(x1_val_seq_lengths), torch.max(x2_val_seq_lengths))

    # loading into DataLoader
    train_dataset = MyDataset(x1_train_tensor, x1_train_seq_lengths, x2_train_tensor, x2_train_seq_lengths, y_train)
    val_dataset = MyDataset(x1_val_tensor, x1_val_seq_lengths, x2_val_tensor, x2_val_seq_lengths, y_val)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
    val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=BATCH_SIZE)

    model = esim.ESIM(embedding=EMBEDDING, embedding_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE, class_num=CLASS_NUM)
    model.to(device=DEVICE)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    best_accuracy = 0.0

    for epoch in range(EPOCH):
        # train
        model.train()
        total_loss = 0.0

        for iterator, (x1, seq_lengths1, x2, seq_lengths2, y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            # gradient cache cleared
            optimizer.zero_grad()
            x1 = x1.to(device=DEVICE)
            seq_lengths1 = seq_lengths1.to(device=DEVICE)
            x2 = x2.to(device=DEVICE)
            seq_lengths2 = seq_lengths2.to(device=DEVICE)
            y = y.to(device=DEVICE)
            loss = model(x1, seq_lengths1, x2, seq_lengths2, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if iterator % 100 == 0 and iterator:
                tqdm.write(f'Round: {epoch=} {iterator=} , Training average loss: {total_loss / iterator}.')

        cur_accuracy = evaluate(model, val_dataloader, val_dataset)
        tqdm.write(f'Round: {epoch=}, Validation accuracy:{cur_accuracy:.5%}')
        if best_accuracy < cur_accuracy:
            best_accuracy = cur_accuracy
            torch.save(model.state_dict(), f'result/epoch_{epoch}_accuracy_{cur_accuracy:.5%}.pt')


if __name__ == '__main__':
    train()
