import torch
from colossalai.context import ParallelMode
from colossalai.constants import OUTPUT_GROUP_3D
from colossalai.global_variables import tensor_parallel_env as tp_env
from colossalai.nn.layer.parallel_3d._utils import get_parallel_mode_from_env
from colossalai.core import global_context as gpc
from colossalai.communication import all_gather, reduce_scatter


def partition_by_tp(val):
    mapping = {
        None: 1,
        "1d": gpc.get_world_size(ParallelMode.TENSOR),
        "2d": tp_env.summa_dim,
        "2.5d": tp_env.tesseract_dim,
        "3d": tp_env.depth_3d,
    }

    assert val % mapping[tp_env.mode] == 0
    return val // mapping[tp_env.mode]


def get_parallel_mode_for_gather():
    mapping = {
        None: None,
        "1d": ParallelMode.TENSOR,
        "2d": ParallelMode.PARALLEL_2D_ROW,
        "2.5d": ParallelMode.PARALLEL_2P5D_ROW,
        "3d": get_parallel_mode_from_env(OUTPUT_GROUP_3D),
    }

    return mapping[tp_env.mode]


class _GatherForwardReduceScatterBackward(torch.autograd.Function):
    """Gather the input from model parallel region and concatenate.
    Args:
        input_: input matrix.
        parallel_mode: parallel mode.
        dim: dimension
    """

    @staticmethod
    def symbolic(graph, input_, dim, parallel_mode):
        return all_gather(input_, dim=dim, parallel_mode=parallel_mode)

    @staticmethod
    def forward(ctx, input_, dim, parallel_mode):
        ctx.mode = parallel_mode
        ctx.dim = dim
        return all_gather(input_, dim=dim, parallel_mode=parallel_mode)

    @staticmethod
    def backward(ctx, grad_output):
        return reduce_scatter(grad_output, dim=ctx.dim, parallel_mode=ctx.mode), None, None


def gather_fwd_reduce_scatter_bwd(tensor, dim, parallel_mode):
    return _GatherForwardReduceScatterBackward.apply(tensor, dim, parallel_mode)
