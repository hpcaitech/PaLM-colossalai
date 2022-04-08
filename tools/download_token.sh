#!/usr/bin/env bash

OUTPUT_PATH=${1:-"./token"}

wget -P $OUTPUT_PATH https://huggingface.co/gpt2/resolve/main/vocab.json
wget -P $OUTPUT_PATH https://huggingface.co/gpt2/resolve/main/merges.txt