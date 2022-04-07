import torch
import os

import colossalai
import torch
from colossalai.core import global_context as gpc
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.trainer import Trainer, hooks
from colossalai.utils import MultiTimer, get_current_device
from data import build_data
from model import build_model, build_loss
from utils import calc_model_size, AutoregressiveWrapper
from colossalai.zero.shard_utils import (BucketTensorShardStrategy, TensorShardStrategy)
from colossalai.zero.sharded_model import ShardedModelV2
from colossalai.zero.sharded_optim import ShardedOptimizerV2
from colossalai.utils.cuda import get_current_device
from colossalai.nn.optimizer import CPUAdam
from colossalai.zero.init_ctx import ZeroInitContext

def train_palm():
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

    print(gpc.config)
    logger = get_dist_logger()
    if hasattr(gpc.config, "LOG_PATH"):
        log_path = gpc.config.LOG_PATH
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        logger.log_to_file(log_path)

    train_dataloader, test_dataloader = build_data(batch_size=2, seq_len=32)
    logger.info("Dataset loaded.", ranks=[0])

    use_zero = hasattr(gpc.config, 'zero')
    if use_zero:
        numel = torch.zeros(1, dtype=torch.int)
        with ZeroInitContext(
            target_device=torch.device(f'cuda:{get_current_device()}'),
            shard_strategy=BucketTensorShardStrategy(),
            shard_param=True,
            rm_torch_payload_on_the_fly=False,
            model_numel_tensor=numel):
            model = build_model()
        numel = numel[0]
        model = ShardedModelV2(
            model,
            BucketTensorShardStrategy(),
            offload_config=dict(device='cpu'),
            use_memory_tracer=False,
            reuse_fp16_shard=True,
        )
        optimizer = CPUAdam(model.parameters(), lr=1e-3)
        optimizer = ShardedOptimizerV2(model,
                                    optimizer,
                                    cpu_offload=True,
                                    initial_scale=2**5,
                                    gpu_margin_mem_ratio=0.8)
    else:
        model = build_model()

        numel, _ = calc_model_size(model)
        model = model.cuda(get_current_device())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()

    logger.info("Optimizer is built.", ranks=[0])

    if numel < 1e9:
        msg = f"{numel / 1e6:.3f} M"
    else:
        msg = f"{numel / 1e9:.3f} B"
    model_mem = torch.cuda.max_memory_allocated(get_current_device()) / 1024**3
    logger.info("Model is built.", ranks=[0])
    logger.info(f"Parameter size = {msg} | Model memory = {model_mem:.3f} GB.", ranks=[0])

    criterion = build_loss()
    logger.info("Loss is built.", ranks=[0])

    data_iter = iter(train_dataloader)

    for i in range(5):
        model.train()

        loss = model(next(data_iter))
        if use_zero:
            model.backward(loss)
        else:
            loss.backward()

        logger(f"training loss: {loss.item()}", ranks = [0])
        optimizer.step()
        optimizer.zero_grad()
    return

    # failed to run the following code.
    # engine, train_dataloader, test_dataloader, _ = colossalai.initialize(
    #     model=model,
    #     optimizer=optimizer,
    #     criterion=criterion,
    #     train_dataloader=train_dataloader,
    #     test_dataloader=test_dataloader,
    # )

    # timer = MultiTimer()

    # trainer = Trainer(engine=engine, logger=logger, timer=timer)

    # hook_list = [
    #     hooks.LogMetricByEpochHook(logger=logger),
    #     hooks.LogMetricByStepHook(),
    #     hooks.LossHook(),
    #     hooks.ThroughputHook(ignored_steps=5),
    #     # hooks.LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=False),
    #     # hooks.TensorboardHook(log_dir='./tb_logs', ranks=[0]),
    #     # hooks.LogMemoryByEpochHook(logger),
    #     # hooks.LogTimingByEpochHook(timer, logger, ignore_num_train_steps=5),
    #     # hooks.SaveCheckpointHook(checkpoint_dir='./ckpt')
    # ]

    # logger.info("Training start.", ranks=[0])
    # trainer.fit(
    #     train_dataloader=train_dataloader,
    #     test_dataloader=test_dataloader,
    #     epochs=gpc.config.NUM_EPOCHS,
    #     max_steps=10,
    #     hooks=hook_list,
    #     return_output_label=False,
    #     display_progress=True,
    # )
    # logger.info("Training complete.", ranks=[0])


if __name__ == "__main__":
    train_palm()