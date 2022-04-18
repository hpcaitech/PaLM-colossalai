from colossalai.amp import AMP_TYPE

SEQ_LENGTH = 2048
BATCH_SIZE = 16
NUM_EPOCHS = 10
WARMUP_EPOCHS = 1

parallel = dict(
    tensor=dict(mode="1d", size=4),
)

model = dict(
    type="palm_8b",
    use_grad_checkpoint=True,
    use_act_offload=False,
)

fp16 = dict(mode=AMP_TYPE.TORCH)

clip_grad_norm = 1.0

LOG_PATH = "./palm_8b_1d/"
