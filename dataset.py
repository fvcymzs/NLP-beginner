from typing import Tuple
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, x1: torch.Tensor, x1_seq_lengths: torch.Tensor, x2: torch.Tensor, x2_seq_lengths: torch.Tensor, y: torch.Tensor):
        assert x1.shape[0] == len(x1_seq_lengths) == x2.shape[0] == len(x2_seq_lengths) == y.shape[0]
        self.x1 = x1
        self.x2 = x2
        self.x1_seq_lengths = x1_seq_lengths
        self.x2_seq_lengths = x2_seq_lengths
        self.y = y
        self.len = y.shape[0]

    # return length
    def __len__(self) -> int:
        return self.len

    # return items
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x1[index], self.x1_seq_lengths[index], self.x2[index], self.x2_seq_lengths[index], self.y[index]
