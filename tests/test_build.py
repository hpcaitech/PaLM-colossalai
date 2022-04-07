import os

import torch
import colossalai
from colossalai.logging import disable_existing_loggers, get_dist_logger
from data import build_data
from model import build_model, build_loss
from utils import AutoregressiveWrapper

def test_build():
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

    logger = get_dist_logger()
    train_dataloader, test_dataloader = build_data(dataset_path=os.environ['DATA'], 
                                                   tokenizer_path=os.environ['TOKENIZER'],
                                                   seq_len=512,
                                                   batch_size=16)
    logger.info("Dataset loaded.", ranks=[0])

    model = build_model()
    model = model(num_tokens=50304, dim=512, depth=8)
    model = AutoregressiveWrapper(model)
    logger.info("Model is built.", ranks=[0])

    loss = build_loss()()
    logger.info("Loss is built.", ranks=[0])

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=0.01,
                                  weight_decay=0.099)

    engine, train_dataloader, test_dataloader, _ = colossalai.initialize(
        model=model,
        optimizer=optimizer,
        criterion=loss,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
    )

    data_iter = iter(train_dataloader)
    batch = next(data_iter)
    input_ids = batch.pop('input_ids').cuda()
    labels = batch.pop('labels').cuda()

    engine.train()
    engine.zero_grad()
    output = engine(input_ids)
    l = engine.criterion(output, labels)
    engine.backward(l)

if __name__ == "__main__":
    test_build()