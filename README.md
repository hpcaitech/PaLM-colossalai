# Pathways Language Model (PaLM) based on PyTorch
A [Colosssal-AI](https://github.com/hpcaitech/ColossalAI) implementation of the model architect of [Pathways Language Model (PaLM): Scaling to 540 Billion Parameters for Breakthrough Performance](https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html).
We take advantage of Colosssal-AI to exploit multiple optimization stategies, e.g. data parallelism, tensor parallelism, mixed precision & ZeRO, to scale the training to mulple-GPUs.

You are very welcome to contribute in any way to help us enhance the usability of this project.

## Preparation
1. Install requirements, e.g. [Colosssal-AI](https://github.com/hpcaitech/ColossalAI), which is a Pytorch-based large-scale model training system with various efficient parallelization techniques.

```
pip install -r requirements.txt
```

2.  Use [HuggingFace datasets](https://github.com/huggingface/datasets) to download Wikitext-2 dataset. The placeholder
`/PATH/TO/DATA` is optional and is `./wiki_dataset` by default.

```bash
python ./tools/download_wiki.py -o </PATH/TO/DATA>
```

3. Download tokenizer files by calling the following command. The place holder `/PATH/TO/TOKENIZER/` is optional and is `./token` by default.

```bash
bash ./tools/download_token.py </PATH/TO/TOKENIZER/>
```

## Usage
1.  Configure your settings in `CONFIG_FILE.py` like below. We also provide some examples in [./configs](./configs/)
```python
SEQ_LENGTH = 512
BATCH_SIZE = 8
NUM_EPOCHS = 10

parallel = dict(
    tensor=dict(mode='1d', size=2),
)

model = dict(type="palm_small")
```


2.  Set dataset & tokenizer paths
```shell
export DATA=</PATH/TO/DATA/>
export TOKENIZER=</PATH/TO/TOKENIZER/>
```

3.  Run
```shell
torchrun --nproc_per_node NUM_GPUS \
    train.py --from_torch --config CONFIG_FILE.py
```

4.  Run With Docker

    Dockerfile is provided in this repository and you can run PaLM in Docker with the following commands.

```bash
# build docker image
docker build -t palm .

# exec training
docker run -ti --gpus all --rm palm \
    torchrun --nproc_per_node NUM_GPUS \
        train.py --from_torch --config CONFIG_FILE.py
```
