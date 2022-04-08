FROM hpcaitech/colossalai:0.1.2-cuda11.3-torch1.10

# install dependencies
RUN cd /workspace \
    && git clone https://github.com/hpcaitech/PaLM-colossalai.git \
    && cd ./PaLM-colossalai \
    && pip install -r requirements.txt

# prepare dataset
RUN cd /workspace \
    && python ./PaLM-colossalai/tools/download_wiki.py \
    && bash ./PaLM-colossalai/tools/download_token.sh

ENV DATA=/workspace/wiki_dataset
ENV TOKENIZER=/workspace/token

WORKDIR /workspace/PaLM-colossalai
