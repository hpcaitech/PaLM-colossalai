import torch
from model.parallel_palm import Parallel_PaLM
from model.loss import PaLMLoss

def build_model():
    return Parallel_PaLM

def build_loss():
    return PaLMLoss

def build_optimizer():
    return torch.optim.Adam

__all__ = ['build_model', 'build_loss', 'build_optimizer']