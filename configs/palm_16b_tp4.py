from colossalai.zero.shard_utils import TensorShardStrategy

SEQ_LENGTH = 2048
BATCH_SIZE = 4
NUM_EPOCHS = 1
# WARMUP_EPOCHS = 1

parallel = dict(
    tensor=dict(mode="1d", size=4),
)

model = dict(
    type="palm_16b",
    use_grad_checkpoint=True,
    use_act_offload=True,
)

