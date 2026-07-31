ARG DEFAULT_BASE=ubuntu:24.04
ARG BASE_IMAGE=${DEFAULT_BASE}
FROM  ${BASE_IMAGE:-$DEFAULT_BASE} AS pytorch_base

WORKDIR /workspace/
ENV MAX_JOBS=128
ENV PYTORCH_ROCM_ARCH="gfx942;gfx950"
ENV ROCM_AMDGPU_TARGETS="gfx942,gfx950"

# Install Ubuntu dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update \
    && apt install -y \
    gfortran \
    git \
    git-lfs \
    ninja-build \
    g++ \
    pkg-config \
    xxd \
    patchelf \
    automake \
    libtool \
    python3-venv \
    python3-dev \
    python3-pip \
    python-is-python3 \
    libegl1-mesa-dev \
    wget \
    sudo \
    flex \
    liblzma-dev \
    ccache \
    libdw1 \
    libdrm-dev \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# Setup python env
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip \
    && pip install \
        pybind11 \
        typeguard \
        wheel==0.45.1 \
        cmake==3.31.6 \
        ninja==1.11.1.3 \
        packaging==25.0 \
        setuptools==75.1.0 \
    && rm -rf /root/.cache

RUN apt update \
    && apt install -y ninja-build autoconf libtool flex rdma-core ffmpeg \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# Workaround for the HSA_STATUS_ERROR_OUT_OF_RESOURCES issue
ENV HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0
ENV HSA_NO_SCRATCH_RECLAIM=1

# Pytorch + rocm
# See: https://rocm.nightlies.amd.com/whl-multi-arch/torch/
ARG PYTORCH_VERSION=2.12.0+rocm7.14.0a20260606

RUN pip install \
    cxxfilt==0.3.0 \
    tqdm==4.67.3 \
    pyyaml==6.0.3 \
    pytest==9.0.3 \
    matplotlib==3.10.9 \
    pandas==2.3.3 \
    py-cpuinfo==9.0.0 \
    build==1.5.0 \
    && rm -rf /root/.cache

RUN python -m pip uninstall -y torch && \
    python -m pip install \
        --index-url https://rocm.nightlies.amd.com/whl-multi-arch \
        --pre \
        torch==${PYTORCH_VERSION} \
        amd-torch-device-gfx942==${PYTORCH_VERSION} \
        amd-torch-device-gfx950==${PYTORCH_VERSION} \
        rocm-sdk-devel \
        rocm-sdk-device-gfx942 \
        rocm-sdk-device-gfx950 \
        torchaudio \
        torchvision \
        amd-torchvision-device-gfx942 \
        amd-torchvision-device-gfx950 \
        apex \
    && rm -rf /root/.cache

RUN rocm-sdk init
ENV ROCM_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel
# Note: Primus uses `ROCM_HOME` instead of `ROCM_PATH`. Set both to avoid potential issues.
ENV ROCM_HOME=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel
ENV HIP_PLATFORM=amd
ENV HIP_PATH=$ROCM_PATH
ENV HIP_CLANG_PATH=$ROCM_PATH/llvm/bin
ENV HIP_INCLUDE_PATH=$ROCM_PATH/include
ENV HIP_LIB_PATH=$ROCM_PATH/lib
ENV HIP_DEVICE_LIB_PATH=$ROCM_PATH/lib/llvm/amdgcn/bitcode
ENV PATH="$ROCM_PATH/bin:$HIP_CLANG_PATH:$PATH"
ENV LD_LIBRARY_PATH="$HIP_LIB_PATH:$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/llvm/lib:$ROCM_PATH/lib/host-math/lib:$ROCM_PATH/lib/rocm_sysdeps/lib"
ENV LIBRARY_PATH="$HIP_LIB_PATH:$ROCM_PATH/lib:$ROCM_PATH/lib64"
ENV CPATH=$HIP_INCLUDE_PATH
ENV PKG_CONFIG_PATH="$ROCM_PATH/lib/pkgconfig"

# FA
ARG FA_REPO=https://github.com/ROCm/flash-attention.git
ARG FA_BRANCH=6387433156558135a998d5568a9d74c1778666d8
ENV GPU_ARCHS="${PYTORCH_ROCM_ARCH}"

RUN git clone --recursive ${FA_REPO} \
    && cd flash-attention \
    && git checkout ${FA_BRANCH} \
    && python setup.py install \
    && cd .. \
    && rm -rf flash-attention \
    && rm -rf /root/.cache

# Latest TransformerEngine
ARG TE_REPO=https://github.com/ROCm/TransformerEngine.git
ARG TE_BRANCH=6ac51951861731cbc87fa87f029c0e6a623113bf

ENV NVTE_USE_HIPBLASLT=1
ENV NVTE_FRAMEWORK=pytorch
ENV NVTE_ROCM_ARCH=${PYTORCH_ROCM_ARCH}
ENV NVTE_USE_CAST_TRANSPOSE_TRITON=1
ENV NVTE_CK_IS_V3_ATOMIC_FP32=0
ENV NVTE_CK_USES_BWD_V3=1
ENV NVTE_CK_USES_FWD_V3=1
ENV CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=2
ENV NVTE_CK_HOW_V3_BF16_CVT=2
ENV NVTE_USE_ROCM=1

RUN git clone --recursive ${TE_REPO} \
    && cd TransformerEngine \
    && git checkout ${TE_BRANCH} \
    && git submodule update --init --recursive \
    && pip install psutil \
    && MAX_JOBS=${MAX_JOBS} pip install --no-build-isolation . \
    && cd .. \
    && rm -rf /root/.cache


# Aiter
ARG AITER_REPO=https://github.com/ROCm/aiter.git
ARG AITER_COMMIT=b5e03ed191fca11ee423226537ef8d9435e432a6

RUN pip uninstall aiter amd-aiter -y \
    && rm -rf aiter \
    && git clone --recursive ${AITER_REPO} \
    && cd aiter \
    && git checkout ${AITER_COMMIT} \
    && git submodule update --init --recursive \
    && PREBUILD_KERNELS=3 pip install --no-cache-dir --use-pep517 . \
    && cd .. \
    && rm -rf aiter


# Some secrets could leak from these files. Explicitly clean them out.
RUN rm -f /workspace/Megatron-LM/.gitlab/stages/00.pre.yml \
 /workspace/Primus/third_party/Megatron-LM/.gitlab/stages/00.pre.yml

# Secret scan will complain those files to be secrets. Explicitly remove them.
RUN rm -f /workspace/Primus/.git/packed-refs /workspace/Primus/.git/modules/third_party/**/packed-refs

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
