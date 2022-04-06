import gzip
import random

import numpy as np
import torch
import torch.optim as optim
import tqdm
from torch.nn import functional as F


from model.palm import PaLM
from model.autoregressive_wrapper import AutoregressiveWrapper
from data_loader import GetTestDataLoader

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 2e-4
VALIDATE_EVERY = 100
GENERATE_EVERY = 500
GENERATE_LENGTH = 512
SEQ_LEN = 1024


def decode_token(token):
    return str(chr(max(32, token)))


def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

if __name__ == '__main__':
    model = PaLM(num_tokens=256, dim=512, depth=8)
    model = AutoregressiveWrapper(model, max_seq_len=2048)
    model.cuda()

    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_dataset, val_dataset, train_loader, val_loader = GetTestDataLoader(BATCH_SIZE, SEQ_LEN)
    for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10.0, desc="training"):
        model.train()

        for __ in range(GRADIENT_ACCUMULATE_EVERY):
            loss = model(next(train_loader))
            loss.backward()

        print(f"training loss: {loss.item()}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()
        optim.zero_grad()

        if i % VALIDATE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                loss = model(next(val_loader))
                print(f"validation loss: {loss.item()}")

        if i % GENERATE_EVERY == 0:
            model.eval()
            inp = random.choice(val_dataset)[:-1]
            prime = decode_tokens(inp)
            print(f"%s \n\n %s", (prime, "*" * 100))

            sample = model.generate(inp[None, ...], GENERATE_LENGTH)
            output_str = decode_tokens(sample[0])
            print(output_str)