FROM frankleeeee/pytorch-cuda:1.10.1-11.3.0

# install dependencies
RUN cd /workspace \
    && git clone https://github.com/hpcaitech/PaLM-colossalai.git \
    && cd ./PaLM-colossalai \
    && pip install -r requirements.txt

# prepare dataset
RUN cd /workspace \
    && python ./PaLM-colossalai/tools/download_wiki.py \
    && bash ./PaLM-colossalai/tools/download_token.py

ENV DATA=/workspace/wiki_dataset
ENV TOKEN=/workspace/token

