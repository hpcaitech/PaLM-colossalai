from colossalai.nn.optimizer import CPUAdam
from torch.optim import Adam
from colossalai.zero.shard_utils import TensorShardStrategy

VOCAB_SIZE = 50304
SEQ_LENGTH = 1024
BATCH_SIZE = 8
NUM_EPOCHS = 10

TENSOR_PARALLEL_SIZE = 2
TENSOR_PARALLEL_MODE = '1d'

zero = dict(
    model_config=dict(
        offload_config=dict(device="cpu"),
        shard_strategy=TensorShardStrategy()
    ),
    optimizer_config=dict(
        cpu_offload=True,
        initial_scale=2**5,
    )
)

# We use a fast CPU Adam here 
# Since we set cpu_offload=True in optimizer_config
optimizer = dict(
    type=CPUAdam,
    lr=0.001,
    weight_decay=1e-2,
)

parallel = dict(
    pipeline=1,
    tensor=dict(mode=TENSOR_PARALLEL_MODE, size=TENSOR_PARALLEL_SIZE),
)

model = "palm_small"
