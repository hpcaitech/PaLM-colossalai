import torch
import torch.distributed as dist
import torch.nn.functional as F
from colossalai.constants import NUM_PARTITIONS
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.utils import get_current_device
from torch import nn


def calc_model_size(model: torch.nn.Module):
    tensor_parallel_size = gpc.tensor_parallel_size
    numel = 0
    numel_per_device = 0
    for p in model.parameters():
        num_partitions = getattr(p, NUM_PARTITIONS, 0)
        if tensor_parallel_size > 1 and num_partitions > 1:
            numel += p.numel() * num_partitions
        else:
            numel += p.numel()
        numel_per_device += p.numel()

    if tensor_parallel_size > 1:
        numel = torch.tensor(numel).to(get_current_device())
        numel = dist.all_reduce(numel, group=gpc.get_group(ParallelMode.TENSOR)) / tensor_parallel_size
        numel = numel.item()

    return numel, numel_per_device


"""Autoregressive wrapper adapted from
https://github.com/lucidrains/PaLM-pytorch/blob/main/palm_pytorch/autoregressive_wrapper.py
"""

# helper function


def exists(val):
    return val is not None


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


# top k filtering


def top_k(logits, thres=0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(1, ind, val)
    return probs


class AutoregressiveWrapper(nn.Module):
    def __init__(self, net, max_seq_len=2048, pad_value=0):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.pad_value = pad_value
        self.net = net

    @torch.no_grad()
    @eval_decorator
    def generate(self, start_tokens, seq_len, eos_token=None, temperature=1.0, filter_thres=0.9, **kwargs):
        b, t, device = *start_tokens.shape, start_tokens.device

        out = start_tokens

        for _ in range(seq_len):
            logits = self.net(out, **kwargs)[:, -1, :]

            filtered_logits = top_k(logits, thres=filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim=-1)

            sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)

            if exists(eos_token):
                is_eos_token = out == eos_token

                if is_eos_token.any(dim=-1).all():
                    # mask out everything after the eos tokens
                    shifted_is_eos_tokens = F.pad(is_eos_token, (1, -1))
                    mask = shifted_is_eos_tokens.float().cumsum(dim=-1) >= 1
                    out = out.masked_fill(mask, self.pad_value)
                    break

        out = out[:, t:]
        return out

    def forward(self, x, **kwargs):
        return self.net(x, **kwargs)
