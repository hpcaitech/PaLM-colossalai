
VOCAB_SIZE = 50304
SEQ_LENGTH = 1024
BATCH_SIZE = 8
NUM_EPOCHS = 10

TENSOR_PARALLEL_SIZE = 2
TENSOR_PARALLEL_MODE = '1d'

parallel = dict(
    pipeline=1,
    tensor=dict(mode=TENSOR_PARALLEL_MODE, size=TENSOR_PARALLEL_SIZE),
)

model = dict(vocab_size=VOCAB_SIZE,
             max_position_embeddings=SEQ_LENGTH)