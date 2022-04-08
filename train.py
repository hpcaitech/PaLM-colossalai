import os
import contextlib

import colossalai
import torch
from colossalai.core import global_context as gpc
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.trainer import Trainer, hooks
from colossalai.utils import MultiTimer, get_current_device
from data import build_data
from model import build_model, build_loss
from utils import calc_model_size, AutoregressiveWrapper
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
    
    use_zero = hasattr(gpc.config, 'zero')
    ctx = contextlib.nullcontext()
    if use_zero:
        ctx = ZeroInitContext(target_device=torch.cuda.current_device(),
                              shard_strategy=gpc.config.zero.model_config.shard_strategy,
                              shard_param=True
                              )
        

    logger = get_dist_logger()
    if hasattr(gpc.config, "LOG_PATH"):
        log_path = gpc.config.LOG_PATH
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        logger.log_to_file(log_path)

    train_dataloader, test_dataloader = build_data(dataset_path=os.environ['DATA'], 
                                                   tokenizer_path=os.environ['TOKENIZER'],
                                                   seq_len=512,
                                                   batch_size=16)
    logger.info("Dataset loaded.", ranks=[0])

    with ctx:
        model = build_model()(num_tokens=50304, dim=512, depth=8)
        model = AutoregressiveWrapper(model)

    '''

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

    optimizer = build_optimizer()

        
    model_mem = torch.cuda.max_memory_allocated(get_current_device()) / 1024**3
    logger.info("Model is built.", ranks=[0])
    '''

    criterion = build_loss()()
    logger.info("Loss is built.", ranks=[0])


    if use_zero:
        optimizer = gpc.config.optimizer.pop('type')(model.parameters(), **gpc.config.optimizer)
    else:
        optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=0.01,
                                  weight_decay=0.099)
    logger.info("Optimizer is built.", ranks=[0])

    engine, train_dataloader, test_dataloader, _ = colossalai.initialize(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader
    )

    def batch_data_process_func(batch_data):
        data = batch_data['input_ids']
        labels = batch_data['labels']
        return data, labels
    engine.schedule.batch_data_process_func = batch_data_process_func

    timer = MultiTimer()
    trainer = Trainer(engine=engine, logger=logger, timer=timer)

    hook_list = [
    ]

    logger.info("Training start.", ranks=[0])
    trainer.fit(
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        epochs=gpc.config.NUM_EPOCHS,
        max_steps=10,
        hooks=hook_list,
        return_output_label=False,
        display_progress=True,
    )
    logger.info("Training complete.", ranks=[0])


if __name__ == "__main__":
    train_palm()
