FROM ubuntu:22.04 AS ubuntu

#
# Install basic packages from OS distro
#
FROM ubuntu AS base

ENV DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.12

RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    apt update && \
    apt install -y git software-properties-common curl rsync dialog gfortran wget sqlite3

RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    if ! python3 --version | grep -q ${PYTHON_VERSION} ; then \
    add-apt-repository -y ppa:deadsnakes/ppa && apt update && \
    apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
                       python${PYTHON_VERSION}-lib2to3 python-is-python3 ; fi

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} && \
    ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION}



RUN wget -nv -O /tmp/cmake-3.26.4-linux-x86_64.tar.gz https://cmake.org/files/v3.26/cmake-3.26.4-linux-x86_64.tar.gz && \
    tar zfx /tmp/cmake-3.26.4-linux-x86_64.tar.gz -C /opt/ && \
    mv /opt/cmake-3.26.4-linux-x86_64 /opt/cmake-3.26.4 && \
    rm -f /tmp/cmake-3.26.4-linux-x86_64.tar.gz

ENV PATH=/opt/cmake-3.26.4/bin:$PATH

#
# Install ROCm rpm packages
#
FROM base AS rocm_deb

ARG ROCM_VERSION=7.2
ARG AMDGPU_VERSION=30.30

RUN curl -sL https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add - \
    && printf "deb [arch=amd64] https://repo.radeon.com/rocm/apt/$ROCM_VERSION/ jammy main\n" | tee /etc/apt/sources.list.d/rocm.list \
    && printf "deb [arch=amd64] https://repo.radeon.com/amdgpu/$AMDGPU_VERSION/ubuntu jammy main\n" | tee /etc/apt/sources.list.d/amdgpu.list \
    && printf "Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n" | tee /etc/apt/preferences.d/rocm-pin-600 \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y rocm && \
    find /opt/rocm/lib -type f -name '*gfx*' | grep -Ev "${GFX_ARCH}" | xargs rm -f && \
    find /opt/rocm/lib/hipblaslt/library -type f -name '*gfx*' | grep -Ev "${GFX_ARCH}" | xargs rm -f && \
    find /opt/rocm/lib/rocblas/library -type f -name '*gfx*' | grep -Ev "${GFX_ARCH}" | xargs rm -f && \
    find /opt/rocm/share/miopen/db -type f -name '*gfx*' | grep -Ev "${GFX_ARCH}" | xargs rm -f && \
    sqlite3 /opt/rocm/lib/rocfft/rocfft_kernel_cache.db "delete from cache_v1 where arch != '${GFX_ARCH}' ; vacuum"

ENV ROCM_HOME=/opt/rocm
ENV CPLUS_INCLUDE_PATH=/opt/rocm/include:
ENV LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH

#
# Install pytorch packages
#
FROM rocm_deb AS rocm_torch

ARG GFX_ARCH=gfx942

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip "setuptools<80" wheel numpy einops ninja && \
    pip install /opt/rocm/share/amd_smi

RUN --mount=type=cache,target=/root/.cache/pip \
    cd /tmp && \
    wget -nv https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torch-2.10.0+rocm7.2.0.lw.gitb6ee5fde-cp312-cp312-linux_x86_64.whl -O torch-2.10.0+rocm7.2.0.lw.gitb6ee5fde-cp312-cp312-linux_x86_64.whl && \
    wget -nv https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/apex-1.10.0+rocm7.2.0.gitef17b699-cp312-cp312-linux_x86_64.whl -O apex-1.10.0+rocm7.2.0.gitef17b699-cp312-cp312-linux_x86_64.whl && \
    wget -nv https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torchaudio-2.10.0+rocm7.2.0.git5047768f-cp312-cp312-linux_x86_64.whl -O torchaudio-2.10.0+rocm7.2.0.git5047768f-cp312-cp312-linux_x86_64.whl && \
    wget -nv https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torchvision-0.25.0+rocm7.2.0.git82df5f59-cp312-cp312-linux_x86_64.whl -O torchvision-0.25.0+rocm7.2.0.git82df5f59-cp312-cp312-linux_x86_64.whl && \
    wget -nv https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/triton-3.6.0+rocm7.2.0.gitba5c1517-cp312-cp312-linux_x86_64.whl -O triton-3.6.0+rocm7.2.0.gitba5c1517-cp312-cp312-linux_x86_64.whl && \
    pip3 install *.whl && \
    rm -f *.whl

ENV PYTORCH_ROCM_ARCH=${GFX_ARCH}

WORKDIR /root

FROM rocm_torch AS new_toolset

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install numpy einops packaging psutil ninja

#
# Build Flash Attention wheel
#
FROM new_toolset AS fa_build

ARG FA_REPO="https://github.com/ROCm/flash-attention"
ARG FA_TAG="9a25eba569317708ae295e396aaac0050b28e52b"

RUN git clone ${FA_REPO} \
 && cd flash-attention \
 && git checkout ${FA_TAG} \
 && git submodule init \
 && git submodule update

RUN cd flash-attention \
 && GPU_ARCHS=gfx942 BUILD_TARGET=rocm MAX_JOBS=$(nproc) python3 setup.py bdist_wheel \
 && mkdir /install && cp dist/*.whl /install

#
# Install Flash Attention
#
FROM new_toolset AS install_fa

RUN --mount=type=bind,from=fa_build,source=/install,target=/tmp/install \
    --mount=type=cache,target=/root/.cache/pip \
    pip install /tmp/install/*.whl

#
# Install TE
#
FROM install_fa AS te

ENV NVTE_USE_HIPBLASLT=1
ENV NVTE_USE_ROCM=1
ENV NVTE_FRAMEWORK=pytorch
ENV NVTE_ROCM_ARCH=gfx942
ENV NVTE_USE_CAST_TRANSPOSE_TRITON=0
ENV NVTE_CK_USES_BWD_V3=1
ENV NVTE_CK_V3_BF16_CVT=2

ARG TE_TAG="15cf65a70f19d71920f3a4647826b4ac92d0fd47"
RUN pip install pybind11 && \
    git clone https://github.com/ROCm/TransformerEngine.git && \
    cd TransformerEngine && git checkout $(TE_TAG) && git submodule update --init --recursive && \
    GPU_ARCHS=gfx942 MAX_JOBS=$(nproc) python3 setup.py install && \
    cd .. && rm -rf TransformerEngine
#
# Install vllm
#
FROM te AS install_vllm

ENV PYTORCH_ROCM_ARCH="gfx942"
RUN pip install setuptools_scm && \
    mkdir /workspace && cd /workspace && \
    ln -sf /opt/rocm/lib/libamdhip64.so /usr/lib/libamdhip64.so && \
    git clone https://github.com/vllm-project/vllm  && \
    cd vllm && pip install -r requirements/rocm.txt && \
    MAX_JOBS=32 python3 setup.py develop --no-deps

FROM install_vllm AS install_verl

ENV CUPY_INSTALL_USE_HIP=1
ENV ROCM_HOME=/opt/rocm
ENV HCC_AMDGPU_TARGET=gfx942
ARG CUPY_TAG="6c4b343ea1960cff41775e03028b99dfa33e2062"
RUN cd /workspace && git clone https://github.com/ROCm/cupy.git  --recursive && \
    pip install Cython && \
    cd cupy && git checkout $(CUPY_TAG) && MAX_JOBS=64 python3 setup.py install
RUN cd /workspace && git clone https://github.com/volcengine/verl.git && \
    cd verl && pip install -e .
RUN pip uninstall cupy-cuda12x -y && rm -rf /workspace/cupy

ENV MIOPEN_DEBUG_CONV_DIRECT=0
RUN apt install vim -y

