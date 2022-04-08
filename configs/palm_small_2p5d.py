from colossalai.amp import AMP_TYPE

SEQ_LENGTH = 512
BATCH_SIZE = 8
NUM_EPOCHS = 1
# WARMUP_EPOCHS = 1

parallel = dict(
    tensor=dict(mode="2.5d", size=4, depth=1),
)

model = dict(
    type="palm_small",
    # use_grad_checkpoint=False,
    # use_act_offload=False,
)

fp16 = dict(mode=AMP_TYPE.NAIVE)

clip_grad_norm = 1.0
