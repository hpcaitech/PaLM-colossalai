import torch
import numpy as np
import colossalai
import random
import string

from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger


def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = "".join(random.choice(letters) for i in range(int(length)))
    return result_str


class RandomTextDataset(torch.utils.data.Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        return {"input_ids": full_seq, "labels": full_seq}

    def __len__(self):
        return self.data.size(0) // self.seq_len


def build_data_from_random(
    dataset_path: str,
    tokenizer_path: str,
    seq_len: int = 512,
    batch_size: int = 8,
):
    logger = get_dist_logger("build_data_from_random")
    logger.info("Building synthetic data ...", ranks=[0])
    random_data = get_random_string(80e5)
    np_rd = np.fromstring(random_data, dtype=np.uint8)
    train_rd, test_rd = np.split(np_rd, [int(70e5)])
    data_train, data_val = torch.from_numpy(train_rd), torch.from_numpy(test_rd)

    train_dataset = RandomTextDataset(data_train, seq_len)
    test_dataset = RandomTextDataset(data_val, seq_len)

    train_dataloader = colossalai.utils.get_dataloader(
        train_dataset,
        seed=1024,
        batch_size=gpc.config.BATCH_SIZE,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )

    test_dataloader = colossalai.utils.get_dataloader(
        test_dataset,
        seed=1024,
        batch_size=gpc.config.BATCH_SIZE,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
    )

    return train_dataloader, test_dataloader
