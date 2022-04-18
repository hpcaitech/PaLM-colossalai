from model.parallel_palm import Parallel_PaLM
from model.loss import PaLMLoss
from colossalai.core import global_context as gpc


def palm_small(use_grad_checkpoint=False, use_act_offload=False):
    return Parallel_PaLM(
        num_tokens=50304,
        dim=768,
        depth=12,
        dim_head=64,
        num_heads=12,
        ff_mult=4,
        use_grad_checkpoint=use_grad_checkpoint,
        use_act_offload=use_act_offload,
    )


def palm_8b(use_grad_checkpoint=True, use_act_offload=False):
    return Parallel_PaLM(
        num_tokens=50304,
        dim=4096,
        depth=32,
        dim_head=256,
        num_heads=16,
        ff_mult=4,
        use_grad_checkpoint=use_grad_checkpoint,
        use_act_offload=use_act_offload,
    )


def palm_16b(use_grad_checkpoint=True, use_act_offload=False):
    return Parallel_PaLM(
        num_tokens=50304,
        dim=4096,
        depth=64,
        dim_head=256,
        num_heads=16,
        ff_mult=4,
        use_grad_checkpoint=use_grad_checkpoint,
        use_act_offload=use_act_offload,
    )


def palm_30b(use_grad_checkpoint=True, use_act_offload=True):
    return Parallel_PaLM(
        num_tokens=50304,
        dim=6144,
        depth=48,
        dim_head=256,
        num_heads=24,
        ff_mult=4,
        use_grad_checkpoint=use_grad_checkpoint,
        use_act_offload=use_act_offload,
    )


def palm_62b(use_grad_checkpoint=True, use_act_offload=True):
    return Parallel_PaLM(
        num_tokens=50304,
        dim=8192,
        depth=64,
        dim_head=256,
        num_heads=32,
        ff_mult=4,
        multi_query=True,
        use_grad_checkpoint=use_grad_checkpoint,
        use_act_offload=use_act_offload,
    )


def build_model():
    assert hasattr(gpc.config, "model") and gpc.config.model.type in [
        "palm_small",
        "palm_8b",
        "palm_16b",
        "palm_30b",
        "palm_62b",
    ], 'Invalid model name. Example usage: model = dict(type="palm_small")'
    model_kwargs = dict()
    if hasattr(gpc.config.model, "use_grad_checkpoint"):
        model_kwargs["use_grad_checkpoint"] = gpc.config.model.use_grad_checkpoint
    if hasattr(gpc.config.model, "use_act_offload"):
        model_kwargs["use_act_offload"] = gpc.config.model.use_act_offload
    return globals()[gpc.config.model.type](**model_kwargs)


def build_loss():
    return PaLMLoss()


__all__ = ["build_model", "build_loss"]
