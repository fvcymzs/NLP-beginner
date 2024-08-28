import os
import torch
from torch.utils.data import DataLoader

import filehandler
from dataset import MyDataset
import esim
import main


DIR = 'result/'
files = os.listdir(DIR)
count = 0
for file in files:
    if count < int(file[6]):
        count = int(file[6])
        name = file
PATH = DIR + name

model = esim.ESIM(embedding=main.EMBEDDING, embedding_size=main.EMBEDDING_SIZE, hidden_size=main.HIDDEN_SIZE, class_num=main.CLASS_NUM)
model.load_state_dict(torch.load(PATH))
model.to(main.DEVICE)

x1_test_list_list_int, x2_test_list_list_int, y_test_orig = filehandler.load_pickle('data/snli_1.0/snli_1.0_test.pkl')

# handle data
x1_test_tensor = torch.nn.utils.rnn.pad_sequence([torch.tensor(sentence, dtype=torch.long, device=None) for sentence in x1_test_list_list_int],
                                                 batch_first=True,
                                                 padding_value=main.VOCAB.pad_index)
x1_test_seq_lengths = torch.tensor([len(sentence) for sentence in x1_test_list_list_int])

x2_test_tensor = torch.nn.utils.rnn.pad_sequence([torch.tensor(sentence, dtype=torch.long, device=None) for sentence in x2_test_list_list_int],
                                                 batch_first=True,
                                                 padding_value=main.VOCAB.pad_index)
x2_test_seq_lengths = torch.tensor([len(sentence) for sentence in x2_test_list_list_int])
y_test = torch.tensor(y_test_orig)
print(x1_test_tensor.shape, x2_test_tensor.shape, torch.max(x1_test_seq_lengths), torch.max(x2_test_seq_lengths))

# loading into DataLoader
test_dataset = MyDataset(x1_test_tensor, x1_test_seq_lengths, x2_test_tensor, x2_test_seq_lengths, y_test)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=main.BATCH_SIZE)
print(main.evaluate(model, test_dataloader, test_dataset))
