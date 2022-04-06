import colossalai
import torch
import torch.multiprocessing as mp
from colossalai.testing import parameterize
from colossalai.utils import free_port
from functools import partial
from model.parallel_palm import ParallelPalmTransformerLayer


@parameterize('multi_query', [True, False])
def run_parallel_transformer_layer(multi_query):
    layer = ParallelPalmTransformerLayer(dim=512, multi_query=multi_query).cuda()
    data = torch.rand(32, 128, 512).cuda()
    out = layer(data)
    loss = out.mean()
    loss.backward()

def run_dist(rank, world_size, config, port):
    colossalai.launch(config=config, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_parallel_transformer_layer()

    

@parameterize('mode', ['1d'])
def test_parallel_palm_transformer_layer(mode):
    world_size = 4

    config = dict(
        parallel=dict(
            tensor=dict(
                size=4,
                mode=mode
            )
        )
    )
    run_func = partial(run_dist, world_size=world_size, config=config, port=free_port())
    mp.spawn(run_func, nprocs=world_size)

if __name__ == '__main__':
    test_parallel_palm_transformer_layer()