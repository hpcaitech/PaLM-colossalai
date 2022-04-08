# PaLM - Pytorch
A Colossal-AI implementation of [PaLM](https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html), which enables multiple optimization stategy, e.g. data parallelism, tensor parallelism & ZeRO, to reduce memory cost.

You are very welcome to contribute in any way to help us enhance the usability of this project.

## Preparation
1. Install [Colosssal-AI](https://github.com/hpcaitech/ColossalAI), which is a Pytorch-based large-scale model training system with various efficient parallelization techniques.

```
pip install colossalai
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
1.  Configure your settings in `CONFIG_FILE.py`, for example
```python
SEQ_LENGTH = 2048
BATCH_SIZE = 8
NUM_EPOCHS = 10

parallel = dict(
    tensor=dict(mode='1d', size=2),
)

model = "palm_small"
```

We have provided some in [./configs](./configs/)
2.  Run
```shell
DATA=/PATH/TO/DATA/ TOKENIZER=/PATH/TO/TOKENIZER/ torchrun --nproc_per_node=NUM_GPUS train.py --from_torch --config CONFIG_FILE.py
```
