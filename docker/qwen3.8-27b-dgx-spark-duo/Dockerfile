FROM vllm/vllm-openai:latest

RUN apt-get update && apt-get install -y \
    rdma-core \
    libibverbs1 \
    ibverbs-utils \
    librdmacm1 \
    libibumad3 \
    build-essential \
    libopenmpi-dev \
    openmpi-bin \
    git \
    && rm -rf /var/lib/apt/lists/*

# docker build -t vllm-openai-rdma:latest .