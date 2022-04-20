from asyncio.log import logger
import contextlib
import os

import colossalai
import torch
from colossalai.core import global_context as gpc
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.trainer import Trainer, hooks
from colossalai.utils import MultiTimer, get_current_device
from colossalai.zero.init_ctx import ZeroInitContext
from colossalai.nn.optimizer import HybridAdam
from colossalai.context import ParallelMode

from data import build_data
from model import build_loss, build_model
from utils import AutoregressiveWrapper, calc_local_model_size, calc_mem
from colossalai.utils import colo_set_process_memory_fraction, colo_device_memory_capacity, colo_set_cpu_memory_capacity


def limit_cuda_memory(size_in_GB: int):
    cuda_capacity = colo_device_memory_capacity(get_current_device())
    if size_in_GB * (1024**3) < cuda_capacity:
        colo_set_process_memory_fraction(size_in_GB * (1024**3) / cuda_capacity)
        logger = get_dist_logger()
        logger.info("Using {} GB of GPU memory".format(size_in_GB))

def limit_cpu_memory(size_in_GB: int):
    colo_set_cpu_memory_capacity(size_in_GB * 1024 ** 3)

def train_palm():
    assert torch.cuda.is_available()
    # limit cuda memory of each GPU to 40GB, if you are using a high-end GPU.
    limit_cuda_memory(40)

    # limit the cpu memory of the CPU to 312 GB
    # limit_cpu_memory(312)

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

    assert hasattr(gpc.config, "BATCH_SIZE"), "Please provide BATCH_SIZE in your configuration"
    assert hasattr(gpc.config, "SEQ_LENGTH"), "Please provide SEQ_LENGTH in your configuration"
    assert hasattr(gpc.config, "NUM_EPOCHS"), "Please provide NUM_EPOCHS in your configuration"

    use_zero = hasattr(gpc.config, "zero")
    ctx = contextlib.nullcontext()
    tflop = 0 
    if use_zero:
        ctx = ZeroInitContext(
            target_device=torch.cuda.current_device(),
            shard_strategy=gpc.config.zero.model_config.shard_strategy,
            shard_param=True,
        )

    logger = get_dist_logger()
    if hasattr(gpc.config, "LOG_PATH"):
        log_path = gpc.config.LOG_PATH
        logger.log_to_file(log_path)

    with ctx:
        model = build_model()
        model = AutoregressiveWrapper(model)
    
    seq_len=gpc.config.SEQ_LENGTH
    batch_size=gpc.config.BATCH_SIZE

    # numel is a model elem in a DP process.
    numel = 0
    if use_zero:
        numel = ctx.model_numel_tensor.item()
    else:
        numel = calc_local_model_size(model)

    # global Tera FLOating Points operations per iteration.
    tflop = numel * batch_size * seq_len \
            * gpc.get_world_size(ParallelMode.MODEL) * gpc.get_world_size(ParallelMode.DATA) * 8 / (1024 ** 4)

    if numel < 1e9:
        msg = f"{numel / 1e6:.3f} M"
    else:
        msg = f"{numel / 1e9:.3f} B"

    model_mem = torch.cuda.max_memory_allocated(get_current_device()) / 1024**3

    logger.info("Model is built.", ranks=[0])
    logger.info(f"Parameter size = {msg} | Model memory = {model_mem:.3f} GB.", ranks=[0])

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
    use_cpu_adam = (
        hasattr(gpc.config, "zero")
        and hasattr(gpc.config.zero, "model_config")
        and getattr(gpc.config.zero.model_config, "tensor_placement_policy") != "cuda"
    )
    optimizer = HybridAdam if use_cpu_adam else torch.optim.AdamW
    optimizer = optimizer(model.parameters(), lr=0.001, weight_decay=1e-2)

    # total_steps = gpc.config.NUM_EPOCHS * len(train_dataloader)
    # warmup_steps = getattr(gpc.config, "WARMUP_EPOCHS", 0) * len(train_dataloader)
    # lr_scheduler = LinearWarmupLR(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)

    logger.info("Optimizer is built.", ranks=[0])

    engine, train_dataloader, _, _ = colossalai.initialize(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        # lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
    )

    def batch_data_process_func(batch_data):
        data = batch_data["input_ids"]
        labels = batch_data["labels"]
        return data, labels

    engine.schedule.batch_data_process_func = batch_data_process_func

    timer = MultiTimer()
    trainer = Trainer(engine=engine, logger=logger, timer=timer)

    hook_list = [
        hooks.LogMetricByEpochHook(logger=logger),
        hooks.LogMetricByStepHook(),
        hooks.LossHook(),
        hooks.ThroughputHook(ignored_steps=10, tflop_per_step = tflop, use_local = False),
        # hooks.LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=False),
        hooks.LogMemoryByEpochHook(logger),
        # hooks.SaveCheckpointHook(checkpoint_dir="./palm.ckpt", model=model),
    ]

    logger.info("Training start.", ranks=[0])
    trainer.fit(
        train_dataloader=train_dataloader,
        epochs=gpc.config.NUM_EPOCHS,
        max_steps=20,
        hooks=hook_list,
        return_output_label=False,
        display_progress=True,
    )

    opt_state = engine.optimizer.state_dict()
    if isinstance(engine.optimizer, colossalai.amp.naive_amp.NaiveAMPOptimizer):
        opt_state = opt_state['optimizer']
    os_mem = calc_mem(opt_state)
    logger.info(f"{engine.optimizer.__class__.__name__} state memory usage = {os_mem / 1024**2:.3f} MB", ranks=[0])

    gpc.destroy()
    logger.info("Training complete.", ranks=[0])


if __name__ == "__main__":
    train_palm()
