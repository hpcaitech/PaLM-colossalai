FROM nvcr.io/nvidia/pytorch:21.07-py3

# install dependencies
RUN pip install -U pip setuptools wheel \
 && pip install pytest tensorboard \
 && pip install colossalai == 0.1.2

