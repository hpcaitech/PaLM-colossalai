from asyncio.log import logger
import contextlib
import os
from packaging import version
from functools import partial
from time import time

import colossalai
import torch
from colossalai.core import global_context as gpc
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.trainer import Trainer, hooks
from colossalai.utils import MultiTimer, get_current_device
from colossalai.zero.init_ctx import ZeroInitContext
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.tensor import ColoParameter, ComputePattern, ComputeSpec, ProcessGroup, ReplicaSpec, ShardSpec
from colossalai.nn.optimizer import HybridAdam
from colossalai.nn.parallel import ZeroDDP
from colossalai.context import ParallelMode
from colossalai.nn.optimizer.gemini_optimizer import GeminiAdamOptimizer

from data import build_data
from model import build_loss, build_model
from utils import AutoregressiveWrapper, calc_local_model_size, calc_mem
from colossalai.utils import colo_set_process_memory_fraction, colo_device_memory_capacity


def limit_cuda_memory(size_in_GB: int):
    cuda_capacity = colo_device_memory_capacity(get_current_device())
    if size_in_GB * (1024**3) < cuda_capacity:
        colo_set_process_memory_fraction(size_in_GB * (1024**3) / cuda_capacity)
        logger = get_dist_logger()
        logger.info("Using {} GB of GPU memory".format(size_in_GB))


def get_tflops(model_numel, batch_size, seq_len, step_time):
    return model_numel * batch_size * seq_len * 8 / 1e12 / (step_time + 1e-12)



# Gemini + ZeRO DDP
def gemini_zero_dpp(model: torch.nn.Module, pg: ProcessGroup, placememt_policy: str = "auto"):
    cai_version = colossalai.__version__
    if version.parse(cai_version) > version.parse("0.1.10"):
        from colossalai.nn.parallel import GeminiDDP
        model = GeminiDDP(model,
                          device=get_current_device(),
                          placement_policy=placememt_policy,
                          pin_memory=True,
                          search_range_mb=32)
    elif version.parse(cai_version) <= version.parse("0.1.10") and version.parse(cai_version) >= version.parse("0.1.9"):
        from colossalai.gemini import ChunkManager, GeminiManager
        chunk_size = ChunkManager.search_chunk_size(model, 64 * 1024**2, 32)
        gemini_manager = GeminiManager(placememt_policy, chunk_manager)
        chunk_manager = ChunkManager(chunk_size,
                                     pg,
                                     enable_distributed_storage=True,
                                     init_device=GeminiManager.get_default_device(placememt_policy))
        model = ZeroDDP(model, gemini_manager)
    else:
        raise NotImplemented(f"CAI version {cai_version} is not supported")
    return model
    
def train_palm():
    assert torch.cuda.is_available()
    
    disable_existing_loggers()
    parser = colossalai.get_default_parser()
    parser.add_argument("--from_torch", default=False, action="store_true")
    args = parser.parse_args()

    if args.from_torch:
        colossalai.launch_from_torch(config=args.config, seed=42)
    else:
        # standard launch
        colossalai.launch(
            config=args.config,
            rank=args.rank,
            world_size=args.world_size,
            local_rank=args.local_rank,
            host=args.host,
            port=args.port,
            seed=42,
        )

    # set to 40GB, if you are using a high-end GPU.
    limit_cuda_memory(40)

    assert hasattr(gpc.config, "BATCH_SIZE"), "Please provide BATCH_SIZE in your configuration"
    assert hasattr(gpc.config, "SEQ_LENGTH"), "Please provide SEQ_LENGTH in your configuration"
    assert hasattr(gpc.config, "NUM_EPOCHS"), "Please provide NUM_EPOCHS in your configuration"

    use_zero = hasattr(gpc.config, "zero")
    #ctx = contextlib.nullcontext()
    tflop = 0 
    default_pg = ProcessGroup(tp_degree=gpc.config.TPDEGREE)
    default_dist_spec = ShardSpec([-1], [gpc.config.TPDEGREE]) if gpc.config.USE_SHARD_INIT else None
    if use_zero:
        # ctx = ZeroInitContext(
        #     target_device=torch.cuda.current_device(),
        #     shard_strategy=gpc.config.zero.model_config.shard_strategy,
        #     shard_param=True,
        # )       
        ctx = ColoInitContext(device='cpu', default_dist_spec=default_dist_spec, default_pg=default_pg)
    with ctx:
            model = build_model()
            model = AutoregressiveWrapper(model)



    logger = get_dist_logger()
    # if hasattr(gpc.config, "LOG_PATH"):
    #     log_path = gpc.config.LOG_PATH
    #     logger.log_to_file(log_path)

    # with ctx:
    #     model = build_model()
    #     model = AutoregressiveWrapper(model)
    
    seq_len=gpc.config.SEQ_LENGTH
    batch_size=gpc.config.BATCH_SIZE

    criterion = build_loss()
    logger.info("Loss is built.", ranks=[0])

    train_dataloader, test_dataloader = build_data(
        dataset_path=os.environ["DATA"],
        tokenizer_path=os.environ["TOKENIZER"],
        seq_len=gpc.config.SEQ_LENGTH,
        batch_size=gpc.config.BATCH_SIZE,
    )

    logger.info("Dataset is loaded.", ranks=[0])

    # We use a fast CPU Adam here
    # If we set cpu_offload=True in optimizer_config
    # use_cpu_adam = (
    #     hasattr(gpc.config, "zero")
    #     and hasattr(gpc.config.zero, "model_config")
    #     and getattr(gpc.config.zero.model_config, "tensor_placement_policy") != "cuda"
    # )
    # optimizer = HybridAdam if use_cpu_adam else torch.optim.AdamW
    # optimizer = optimizer(model.parameters(), lr=0.001, weight_decay=1e-2)
    pg = default_pg
    # Tensor Parallelism (TP)
    #tensor_parallelize(model, pg)
    # Gemini + ZeRO DP, Note it must be used after TP
    model = gemini_zero_dpp(model, pg, gpc.config.placement)

    # build optimizer
    optimizer = GeminiAdamOptimizer(model, lr=1e-3, initial_scale=2**5)

    # total_steps = gpc.config.NUM_EPOCHS * len(train_dataloader)
    # warmup_steps = getattr(gpc.config, "WARMUP_EPOCHS", 0) * len(train_dataloader)
    # lr_scheduler = LinearWarmupLR(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)

    logger.info("Optimizer is built.", ranks=[0])

    #logger.info(get_mem_info(prefix='After init model, '), ranks=[0])

        #numel is a model elem in a DP process.
    numel = 0
    if use_zero:
        #numel = ctx.model_numel_tensor.item()
        numel = sum([p.numel() for p in model.parameters()])
    else:
        numel = calc_local_model_size(model)

    tflop = numel * batch_size * seq_len \
            * gpc.get_world_size(ParallelMode.MODEL) * gpc.get_world_size(ParallelMode.DATA) * 8 / (1024 ** 4)

    get_tflops_func = partial(get_tflops, numel, batch_size, seq_len)

    if numel < 1e9:
        msg = f"{numel / 1e6:.3f} M"
    else:
        msg = f"{numel / 1e9:.3f} B"

    model_mem = torch.cuda.max_memory_allocated(get_current_device()) / 1024**3

    logger.info("Model is built.", ranks=[0])
    logger.info(f"Parameter size = {msg} | Model memory = {model_mem:.3f} GB.", ranks=[0])
     
    torch.cuda.synchronize()
    model.train()
    for n in range(gpc.config.NUM_EPOCHS):
        for step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()
            optimizer.zero_grad()
            start = time()
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            #logger.info(get_mem_info(prefix=f'[{n+1}/{gpc.config.NUM_EPOCHS}] Forward '), ranks=[0])

            optimizer.backward(loss)

            #logger.info(get_mem_info(prefix=f'[{n+1}/{gpc.config.NUM_EPOCHS}] Backward '), ranks=[0])
            optimizer.step()
            #logger.info(get_mem_info(prefix=f'[{n+1}/{gpc.config.NUM_EPOCHS}] Optimizer step '), ranks=[0])
            step_time = time() - start
            logger.info(
                f'[{n+1}/{gpc.config.NUM_EPOCHS}] Loss:{loss.item():.3f}, Step time: {step_time:.3f}s, TFLOPS: {get_tflops_func(step_time):.3f}',
                ranks=[0])

    torch.cuda.synchronize()


if __name__ == "__main__":
    train_palm()
