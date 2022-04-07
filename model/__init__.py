from .palm import PaLM
from .autoregressive_wrapper import AutoregressiveWrapper

def build_model():
    return AutoregressiveWrapper(PaLM(num_tokens=256, dim=512, depth=8) , max_seq_len=2048)


def build_loss():
    return None