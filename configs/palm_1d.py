SEQ_LENGTH = 2048
BATCH_SIZE = 8
NUM_EPOCHS = 10

TENSOR_PARALLEL_SIZE = 4

TENSOR_PARALLEL_MODE = '1d'

parallel = dict(
    pipeline=1,
    tensor=dict(mode=TENSOR_PARALLEL_MODE, size=TENSOR_PARALLEL_SIZE),
)

model = "palm_small"
