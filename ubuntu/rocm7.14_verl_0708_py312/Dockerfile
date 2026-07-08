# Primus Training Dockerfile for ROCm on TheRock
ARG DEFAULT_BASE=rocm/primus:v26.4
ARG BASE_IMAGE=${DEFAULT_BASE}
FROM  ${BASE_IMAGE:-$DEFAULT_BASE} AS pytorch_base



FROM pytorch_base AS install_vllm

ARG VLLM_TAG="18d87a87dc39c1a50f452b218fdcc3aeec48d18e"
RUN pip install setuptools_scm && \
    cd /workspace && \
    ln -sf /opt/rocm/lib/libamdhip64.so /usr/lib/libamdhip64.so && \
    git clone https://github.com/vllm-project/vllm && \
    cd vllm && git checkout ${VLLM_TAG} && pip install -r requirements/rocm.txt && \
    MAX_JOBS=64 python3 setup.py develop --no-deps
RUN pip uninstall tilelang -y && pip install xgrammar==0.1.32
# VLLM patch
RUN sed -i 's/return w13_weight\.contiguous(), w2_weight\.contiguous()/return w13_weight, w2_weight/' \
      /workspace/vllm/vllm/model_executor/layers/fused_moe/oracle/unquantized.py

# Install Causal-Conv1d and its dependencies
ARG CAUSAL_CONV1D_REPO=https://github.com/Dao-AILab/causal-conv1d
ARG CAUSAL_CONV1D_BRANCH=e940ead2fd962c56854455017541384909ca669f

ENV CAUSAL_CONV1D_FORCE_BUILD=TRUE
ENV MAMBA_FORCE_BUILD=TRUE
ENV HIP_ARCHITECTURES=gfx942,gfx950

FROM install_vllm AS install_verl
RUN cd /workspace && git clone https://github.com/verl-project/verl.git && \
    cd verl && pip install .

RUN git clone ${CAUSAL_CONV1D_REPO} causal-conv1d \
    && cd causal-conv1d \
    && git checkout ${CAUSAL_CONV1D_BRANCH} \
    && git show --oneline -s \
    && pip install --no-build-isolation . \
    && cd .. \
    && rm -fr causal-conv1d \
    && rm -rf /root/.cache

# for qwen3.5
RUN pip install -U git+https://github.com/ISEEKYAN/mbridge.git && \
    pip install mcore_bridge --upgrade && \
    pip install megatron-core --upgrade && \
    pip install mathruler && \
    pip install qwen_vl_utils && \
    pip install datasets --upgrade && \
    pip install accelerate --upgrade && \
    pip install TransferQueue && \
    pip install flash-linear-attention

RUN pip install cupy-rocm-7-0
ENV NVTE_FLASH_ATTN=1
ENV NVTE_USE_GROUPED_GEMM_TRITON=1
WORKDIR /workspace/
RUN apt-get update && apt install vim -y

ENV NCCL_DEBUG=ERROR
RUN SP="$(python -c 'import site; print(site.getsitepackages()[0])')"; \
    ln -sfn "$SP/_rocm_sdk_core/share/amd_smi/amdsmi" "$SP/amdsmi"; 


# Clean cache
RUN rm -rf /root/.cache

# Clean build files
RUN rm -rf /*.whl \
    && rm -rf /root/.triton \
    && rm -rf /workspace/aiter/aiter/jit/build \
    && rm -rf /workspace/TransformerEngine

# Training docker manifest
ARG GIT_COMMIT_TAG=DEV
ARG DOCKERFILE_PATH=Dockerfile

# Copy patch files for training docker versioning
RUN mkdir -p /workspace/.manifest \
    && env > /workspace/.manifest/env.txt \
    && pip list > /workspace/.manifest/requirements.txt \
    && dpkg -l > /workspace/.manifest/dpkg-list.txt \
    && echo "${GIT_COMMIT_TAG}" > /workspace/.manifest/training_docker_version

COPY ${DOCKERFILE_PATH} /workspace/.manifest/Dockerfile

CMD ["/usr/bin/bash"]
