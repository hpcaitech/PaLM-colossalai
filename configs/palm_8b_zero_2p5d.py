from colossalai.zero.shard_utils import TensorShardStrategy

SEQ_LENGTH = 2048
BATCH_SIZE = 64
NUM_EPOCHS = 10
WARMUP_EPOCHS = 1

parallel = dict(
    tensor=dict(mode="2.5d", size=4, depth=1),
)

model = dict(
    type="palm_8b",
    use_grad_checkpoint=True,
    use_act_offload=False,
)

zero = dict(
    model_config=dict(
        tensor_placement_policy="cpu",
        shard_strategy=TensorShardStrategy()
    ),
    optimizer_config=dict(
        initial_scale=2**5,
    )
)

clip_grad_norm = 1.0

LOG_PATH = "./palm_8b_zero_2p5d/"
