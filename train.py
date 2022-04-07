import random

import torch
import torch.optim as optim
import tqdm

import colossalai
from colossalai.nn.optimizer import CPUAdam
from colossalai.zero.init_ctx import ZeroInitContext

from model.palm import PaLM
from model.autoregressive_wrapper import AutoregressiveWrapper
from data_loader import GetTestDataLoader
from colossalai.zero.shard_utils import (BucketTensorShardStrategy, TensorShardStrategy)
from colossalai.zero.sharded_model import ShardedModelV2
from colossalai.zero.sharded_optim import ShardedOptimizerV2
from colossalai.utils.cuda import get_current_device

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 1
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

    use_zero = True
    if use_zero:
        colossalai.launch_from_torch(config={})
        with ZeroInitContext(
            target_device=torch.device(f'cuda:{get_current_device()}'),
            shard_strategy=BucketTensorShardStrategy(),
            shard_param=True,
            rm_torch_payload_on_the_fly=False):
            model = PaLM(num_tokens=256, dim=512, depth=8)
        model = ShardedModelV2(
            model,
            BucketTensorShardStrategy(),
            offload_config=dict(device='cpu'),
            use_memory_tracer=True,
            reuse_fp16_shard=True,
        )
        optim = CPUAdam(model.parameters(), lr=LEARNING_RATE)
        optim = ShardedOptimizerV2(model,
                                    optim,
                                    cpu_offload=True,
                                    initial_scale=2**5,
                                    gpu_margin_mem_ratio=0.8)
    else:
        model = PaLM(num_tokens=256, dim=512, depth=8)
        model = AutoregressiveWrapper(model, max_seq_len=2048)
        model.cuda()

        optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_dataset, val_dataset, train_loader, val_loader = GetTestDataLoader(BATCH_SIZE, SEQ_LEN)
    for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10.0, desc="training"):
        model.train()

        for __ in range(GRADIENT_ACCUMULATE_EVERY):
            loss = model(next(train_loader))
            if use_zero:
                model.backward(loss)
            else:
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