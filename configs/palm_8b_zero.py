from colossalai.zero.shard_utils import TensorShardStrategy

SEQ_LENGTH = 2048
BATCH_SIZE = 4
NUM_EPOCHS = 1
# WARMUP_EPOCHS = 1

parallel = dict(
    # tensor=dict(mode="3d", size=8),
)

model = dict(
    type="palm_8b",
    use_grad_checkpoint=True,
    use_act_offload=False,
)

zero = dict(
    model_config=dict(
        shard_strategy=TensorShardStrategy(),
        tensor_placement_policy='cuda',
    ),
    optimizer_config=dict(
        gpu_margin_mem_ratio = 0.8,
        initial_scale=2**5,
    )
)

