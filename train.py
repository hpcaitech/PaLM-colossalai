import contextlib
import os

import colossalai
import torch
from colossalai.core import global_context as gpc
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn import LinearWarmupLR, CPUAdam
from colossalai.trainer import Trainer, hooks
from colossalai.utils import MultiTimer, get_current_device
from colossalai.zero.init_ctx import ZeroInitContext
from colossalai.nn.optimizer import HybridAdam
from colossalai.context import ParallelMode

from data import build_data
from model import build_loss, build_model
from utils import AutoregressiveWrapper, calc_model_size


def train_palm():
    torch.cuda.is_available()
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
    if use_zero:
        seq_len=gpc.config.SEQ_LENGTH
        batch_size=gpc.config.BATCH_SIZE
        tflop =  ctx.model_numel_tensor.item() * batch_size * seq_len * gpc.get_world_size(ParallelMode.DATA) * 8 / (1024 ** 4)

    numel, _ = calc_model_size(model)
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
        and hasattr(gpc.config.zero, "optimizer_config")
        and getattr(gpc.config.zero.optimizer_config, "cpu_offload", False)
    )
    optimizer = HybridAdam if use_cpu_adam else torch.optim.AdamW
    optimizer = optimizer(model.parameters(), lr=0.001, weight_decay=1e-2)

    # total_steps = gpc.config.NUM_EPOCHS * len(train_dataloader)
    # warmup_steps = getattr(gpc.config, "WARMUP_EPOCHS", 0) * len(train_dataloader)
    # lr_scheduler = LinearWarmupLR(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)

    logger.info("Optimizer is built.", ranks=[0])

    engine, train_dataloader, test_dataloader, _ = colossalai.initialize(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        # lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
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
        hooks.ThroughputHook(tflop_per_step = tflop),
        # hooks.LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=False),
        hooks.LogMemoryByEpochHook(logger),
        # hooks.LogTimingByEpochHook(timer, logger, ignore_num_train_steps=5),
        # hooks.SaveCheckpointHook(checkpoint_dir="./palm.ckpt"),
    ]

    logger.info("Training start.", ranks=[0])
    trainer.fit(
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        epochs=gpc.config.NUM_EPOCHS,
        # max_steps=10,
        hooks=hook_list,
        return_output_label=False,
        display_progress=True,
    )

    gpc.destroy()
    logger.info("Training complete.", ranks=[0])


if __name__ == "__main__":
    train_palm()
