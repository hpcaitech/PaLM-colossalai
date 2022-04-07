from model.parallel_palm import Parallel_PaLM
from model.loss import PaLMLoss

def build_model():
    return Parallel_PaLM

def build_loss():
    return PaLMLoss

__all__ = ['build_model', 'build_loss']