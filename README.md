# PaLM - Pytorch
A Colossal-AI implementation of [PaLM](https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html), which enables hybrid optimization techniques, e.g. tensor parallelism & ZeRO, to reduce memory cost.

You are very welcome to contribute in any way to help us enhance the usability of this project.

## Preparation
1.  Use [HuggingFace datasets](https://github.com/huggingface/datasets) to download Wikitext-2 dataset
```python
from datasets import load_dataset
wikitext_dataset = load_dataset('wikitext', 'wikitext-2-v1')
```
2.  Save dataset to `/PATH/TO/DATA/`
```python
wikitext_dataset.save_to_disk('/PATH/TO/DATA/')
```
3. Download tokenizer files to `/PATH/TO/TOKENIZER/`
```shell
wget -P /PATH/TO/TOKENIZER/ https://huggingface.co/gpt2/resolve/main/vocab.json
wget -P /PATH/TO/TOKENIZER/ https://huggingface.co/gpt2/resolve/main/merges.txt
```

## Usage
Configure your settings in `CONFIG_FILE.py`, and run
```shell
DATA=/PATH/TO/DATA/ TOKENIZER=/PATH/TO/TOKENIZER/ torchrun --nproc_per_node=NUM_GPUS train.py --from_torch --config CONFIG_FILE.py
```
