from colossalai.nn.optimizer import CPUAdam
from colossalai.zero.shard_utils import TensorShardStrategy
from model.palm import PaLM
from torch.optim import Adam

BATCH_SIZE = 2
NUM_EPOCHS = 60
SEQ_LEN = 1024


# zero = dict(
#     model_config=dict(
#         offload_config=dict(device="cpu"),
#         shard_strategy=TensorShardStrategy()
#     ),
#     optimizer_config=dict(
#         cpu_offload=True,
#     )
# )
