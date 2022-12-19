export TOKENIZER=./token
export DATA=./wiki_dataset
env OMP_NUM_THREADS=12 torchrun  --nproc_per_node 4  --master_port 29501  train.py --from_torch --config ./configs/palm_30b_2d.py 