FROM ti-acc.tencentcloudcr.com/tiacc/megatron:centos-stream9-20250509-base_tc_repo AS centos9

RUN sed -i.bak -e 's/^baseurl=/# baseurl=/g'  -e 's/^# metalink=/metalink=/g' /etc/yum.repos.d/centos.repo

#
# Install basic packages from OS distro
#
FROM centos9 AS base

ARG PYTHON_VER=3.12

RUN --mount=target=/var/cache/dnf,type=cache,sharing=locked \
    --mount=target=/var/cache/yum,type=cache,sharing=locked \
    dnf config-manager --set-enabled crb && \
    yum install -y --allowerasing rsync sqlite wget git curl sqlite-devel ocl-icd-devel && \
    yum groupinstall -y "Development Tools"

RUN --mount=target=/var/cache/dnf,type=cache,sharing=locked \
    --mount=target=/var/cache/yum,type=cache,sharing=locked \
    yum install -y python3.12-devel

RUN alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VER} 1 && \
    alternatives --set python3 /usr/bin/python${PYTHON_VER} && \
    alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VER} 1 && \
    alternatives --set python /usr/bin/python${PYTHON_VER} && \
    ln -sf /usr/bin/python${PYTHON_VER}-config /usr/bin/python3-config

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python

RUN wget -nv -O /tmp/cmake-3.26.4-linux-x86_64.tar.gz https://cmake.org/files/v3.26/cmake-3.26.4-linux-x86_64.tar.gz && \
    tar zfx /tmp/cmake-3.26.4-linux-x86_64.tar.gz -C /opt/ && \
    mv /opt/cmake-3.26.4-linux-x86_64 /opt/cmake-3.26.4 && \
    rm -f /tmp/cmake-3.26.4-linux-x86_64.tar.gz

ENV PATH=/opt/cmake-3.26.4/bin:$PATH

#
# Install ROCm rpm packages
#
FROM base AS rocm_rpm

ARG AMD_BUILD_ID=2226275
ARG ROCM_BUILD_ID=compute-rocm-rel-7.0/56
ARG GFX_ARCH=gfx942

RUN --mount=target=/var/cache/dnf,type=cache,sharing=locked \
    --mount=target=/var/cache/yum,type=cache,sharing=locked \
    printf "[amdgpu]\nname=AMDGPU ${AMD_BUILD_ID} repository\nbaseurl=https://repo.radeon.com/amdgpu/30.10.2/el/9.6/main/x86_64\nenabled=1\ngpgcheck=0" | tee /etc/yum.repos.d/amdgpu-build.repo && \
    printf "[rocm]\nname=ROCm ${ROCM_BUILD_ID} repository\nbaseurl=https://repo.radeon.com/rocm/el9/7.0.2/main\nenabled=1\ngpgcheck=0\npriority=50" | tee /etc/yum.repos.d/rocm-build.repo && \
    yum install -y rocm && \
	rm -f /etc/yum.repos.d/amdgpu-build.repo && \
    rm -f /etc/yum.repos.d/rocm-build.repo && \
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
FROM rocm_rpm AS rocm_torch

ARG GFX_ARCH=gfx942

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip "setuptools<80" wheel numpy einops ninja && \
    pip install /opt/rocm/share/amd_smi

RUN --mount=type=cache,target=/root/.cache/pip \
    cd /tmp && \
    wget -nv https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0.2/torch-2.8.0+rocm7.0.2.lw.git2cd73af9-cp312-cp312-linux_x86_64.whl -O torch-2.8.0+rocm7.0.2.lw.git2cd73af9-cp312-cp312-linux_x86_64.whl && \
    wget -nv https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0.2/apex-1.8.0a0+rocm7.0.2.git3f26640c-cp312-cp312-linux_x86_64.whl -O apex-1.8.0a0+rocm7.0.2.git3f26640c-cp312-cp312-linux_x86_64.whl && \
    wget -nv https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0.2/torchaudio-2.8.0+rocm7.0.2.git6e1c7fe9-cp312-cp312-linux_x86_64.whl -O torchaudio-2.8.0+rocm7.0.2.git6e1c7fe9-cp312-cp312-linux_x86_64.whl && \
    wget -nv https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0.2/torchvision-0.23.0+rocm7.0.2.git824e8c87-cp312-cp312-linux_x86_64.whl -O torchvision-0.23.0+rocm7.0.2.git824e8c87-cp312-cp312-linux_x86_64.whl && \
    wget -nv https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0.2/triton-3.4.0+rocm7.0.2.gitf9e5bf54-cp312-cp312-linux_x86_64.whl -O triton-3.4.0+rocm7.0.2.gitf9e5bf54-cp312-cp312-linux_x86_64.whl && \
    pip install *.whl && \
    rm -f *.whl

ENV PYTORCH_ROCM_ARCH=${GFX_ARCH}

WORKDIR /root

#
# Install gcc12 for FA and TE build
#
FROM rocm_torch AS new_toolset

RUN --mount=target=/var/cache/dnf,type=cache,sharing=locked \
    --mount=target=/var/cache/yum,type=cache,sharing=locked \
    yum install -y gcc-toolset-12

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install numpy einops packaging psutil ninja

#
# Build Flash Attention wheel
#
FROM new_toolset AS fa_build

ARG FA_REPO="https://github.com/ROCm/flash-attention"
ARG FA_TAG="04adaf0e9028d4bec7073f69e4dfa3f6d3357189"

RUN git clone ${FA_REPO} \
 && cd flash-attention \
 && git checkout ${FA_TAG} \
 && git submodule init \
 && git submodule update

RUN cd flash-attention \
 && GPU_ARCHS=gfx942 BUILD_TARGET=rocm MAX_JOBS=$(nproc) scl enable gcc-toolset-12 'python3 setup.py bdist_wheel' \
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

ARG TE_TAG="91876593ff7f721e06f5d7197aac49d3804b3479"
RUN git clone https://github.com/ROCm/TransformerEngine.git && \
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
    git clone https://github.com/aaab8b/vllm -b a8e7c3e96_sleep_mode && \
    cd vllm && pip install -r requirements/rocm.txt && \
    MAX_JOBS=32 python3 setup.py develop --no-deps

FROM install_vllm AS install_verl

ENV CUPY_INSTALL_USE_HIP=1
ENV ROCM_HOME=/opt/rocm
ENV HCC_AMDGPU_TARGET=gfx942
RUN cd /workspace && git clone https://github.com/ROCm/cupy.git -b enable_rocm7 --recursive && \
    pip install Cython && \
    cd cupy && MAX_JOBS=64 python3 setup.py install && \
    cd /workspace && git clone https://github.com/volcengine/verl.git && \
    cd verl && pip install -e . 
RUN pip uninstall cupy-cuda12x -y && rm -rf /workspace/cupy


#
# Cleanup
#
FROM install_verl AS final_base

RUN mv /etc/yum.repos.d/centos.repo.bak /etc/yum.repos.d/centos.repo

CMD ["scl", "enable", "gcc-toolset-12", "bash"]
