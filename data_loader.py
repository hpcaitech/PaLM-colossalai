from typing import Any
import torch
import gzip
import numpy as np
from torch.utils.data import DataLoader, Dataset

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        return full_seq.cuda()

    def __len__(self):
        return self.data.size(0) // self.seq_len
        
def GetTestDataLoader(batch_size: int, seq_len: int):
    def cycle(loader):
        while True:
            for data in loader:
                yield data
    with gzip.open("./data/enwik8.gz") as file:
        X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
        trX, vaX = np.split(X, [int(90e6)])
        data_train, data_val = torch.from_numpy(trX), torch.from_numpy(vaX)

    train_dataset = TextSamplerDataset(data_train, seq_len)
    val_dataset = TextSamplerDataset(data_val, seq_len)
    train_loader = cycle(DataLoader(train_dataset, batch_size=batch_size))
    val_loader = cycle(DataLoader(val_dataset, batch_size=batch_size))

    return train_dataset, val_dataset, train_loader, val_loader