import colossalai
import torch
import torch.multiprocessing as mp
from colossalai.testing import parameterize
from colossalai.utils import free_port
from functools import partial
from model.parallel_palm import ParallelPalmTransformerLayer
from colossalai.global_variables import tensor_parallel_env as tp_env


@parameterize('multi_query', [True, False])
def run_parallel_transformer_layer(multi_query):
    batch_size = 8
    seq_length = 256
    dim = 384

    layer = ParallelPalmTransformerLayer(dim=dim, multi_query=multi_query).cuda()

    if tp_env.mode == '2d':
        batch_size //= tp_env.summa_dim
        dim //= tp_env.summa_dim
    elif tp_env.mode == '2.5d':
        batch_size //= tp_env.tesseract_dim
        dim //= tp_env.tesseract_dim
    elif tp_env.mode == '3d':
        batch_size //= tp_env.depth_3d ** 2
        dim //= tp_env.depth_3d

    data = torch.rand(batch_size, seq_length, dim).cuda()
    out = layer(data)
    loss = out.mean()
    loss.backward()

def run_dist(rank, world_size, config, port):
    colossalai.launch(config=config, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_parallel_transformer_layer()

    

@parameterize('mode', [None, '1d', '2d', '2.5d', '3d'])
def test_parallel_palm_transformer_layer(mode):
    if mode in ['2.5d', '3d']:
        world_size = 8
    else:
        world_size = 4

    config = dict(
        parallel=dict(
            tensor=dict(
                size=world_size,
                mode=mode
            )
        )
    )

    if mode == '2.5d':
        config['parallel']['tensor']['depth'] = 2

    run_func = partial(run_dist, world_size=world_size, config=config, port=free_port())
    mp.spawn(run_func, nprocs=world_size)

if __name__ == '__main__':
    test_parallel_palm_transformer_layer()