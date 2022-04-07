export TOKENIZER=/data/scratch/huggingface/tokenizers/gpt2/gpt2
export DATA=/data/scratch/huggingface/datasets/wikitext/wikitext-2

torchrun --standalone --nproc_per_node=1 train.py --from_torch

# --config=./configs/zero.py 
