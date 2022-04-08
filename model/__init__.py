from model.parallel_palm import Parallel_PaLM
from model.loss import PaLMLoss
from colossalai.core import global_context as gpc


def palm_small():
    return Parallel_PaLM(num_tokens=50304, dim=768, depth=12, dim_head=64, heads=12, ff_mult=4)


def palm_8b():
    return Parallel_PaLM(num_tokens=50304, dim=4096, depth=32, dim_head=256, heads=16, ff_mult=4)


def palm_62b():
    return Parallel_PaLM(num_tokens=50304, dim=8192, depth=64, dim_head=256, heads=32, ff_mult=4)


def build_model():
    assert hasattr(gpc.config, "model") and gpc.config.model in [
        "palm_small",
        "palm_8b",
        "palm_62b",
    ], 'Invalid model name. Example usage: model = "palm_small"'
    return globals()[gpc.config.model]()


def build_loss():
    return PaLMLoss()


__all__ = ["build_model", "build_loss"]
