ARG DEFAULT_BASE=rocm/primus:v26.4
ARG BASE_IMAGE=${DEFAULT_BASE}
FROM ${BASE_IMAGE:-$DEFAULT_BASE} AS pytorch_base

FROM pytorch_base AS install_vllm
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# Set the working dir explicitly so later RUN steps don't rely on the base
# image's inherited WORKDIR being /workspace.
WORKDIR /workspace
ENV PYTORCH_ROCM_ARCH="gfx942;gfx950"

# Install Causal-Conv1d and its dependencies
ARG CAUSAL_CONV1D_REPO=https://github.com/Dao-AILab/causal-conv1d
ARG CAUSAL_CONV1D_BRANCH=e940ead2fd962c56854455017541384909ca669f

ENV CAUSAL_CONV1D_FORCE_BUILD=TRUE
ENV MAMBA_FORCE_BUILD=TRUE
ENV HIP_ARCHITECTURES=gfx942,gfx950

RUN cd /workspace \
    && git clone ${CAUSAL_CONV1D_REPO} causal-conv1d \
    && cd causal-conv1d \
    && git checkout ${CAUSAL_CONV1D_BRANCH} \
    && git show --oneline -s \
    && pip install --no-cache-dir --no-build-isolation . \
    && cd .. \
    && rm -fr causal-conv1d

ARG VLLM_TAG="18d87a87dc39c1a50f452b218fdcc3aeec48d18e"
RUN pip install --no-cache-dir setuptools_scm && \
    cd /workspace && git clone https://github.com/vllm-project/vllm && \
    cd vllm && git checkout ${VLLM_TAG} && \
    sed -i 's|\(  // std::cout << "create_and_map:\)|#ifdef USE_ROCM\n  // ROCm: zero freshly mapped VRAM to avoid stale data (PR #44972)\n  CUDA_CHECK((CUresult)hipMemset(reinterpret_cast<void*>(d_mem), 0, size));\n  if (error_code != 0) {\n    return;\n  }\n#endif\n\1|' csrc/cumem_allocator.cpp && \
    sed -i 's/layer\.w13_weight\.data = self\._maybe_pad_weight(layer\.w13_weight\.data)/layer.w13_weight.data.copy_(self._maybe_pad_weight(layer.w13_weight.data))/' \
      /workspace/vllm/vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py && \
    sed -i 's/layer\.w2_weight\.data = self\._maybe_pad_weight(layer\.w2_weight\.data)/layer.w2_weight.data.copy_(self._maybe_pad_weight(layer.w2_weight.data))/' \
      /workspace/vllm/vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py && \
    sed -i 's/return w13_weight\.contiguous(), w2_weight\.contiguous()/return w13_weight, w2_weight/' \
      /workspace/vllm/vllm/model_executor/layers/fused_moe/oracle/unquantized.py && \
    pip install --no-cache-dir -r requirements/rocm.txt && \
    MAX_JOBS=64 python3 setup.py develop --no-deps
RUN pip uninstall tilelang -y && pip install --no-cache-dir --no-deps xgrammar==0.1.32


FROM install_vllm AS install_verl
ARG VERL_TAG="7b19203c52503760f92e0368067c7eb744236274"
RUN cd /workspace && git clone https://github.com/verl-project/verl.git && \
    cd verl && git checkout ${VERL_TAG} && \
    git submodule update --init --recursive && \
    sed -i 's/^  vllm: {}/  vllm:\n    disable_custom_all_reduce: True/' /workspace/verl/verl/trainer/config/rollout/rollout.yaml && \
    sed -i 's/^  sglang: {}/  sglang:\n    disable_custom_all_reduce: True\n    attention_backend: triton/' /workspace/verl/verl/trainer/config/rollout/rollout.yaml && \
    pip install --no-cache-dir . 
    
# for qwen3.5 — versions pinned to what was validated on this image for reproducibility.
RUN pip install --no-cache-dir \
    "git+https://github.com/ISEEKYAN/mbridge.git@a61943d7fcb34a190471cfeb0a0eb8bbda621ddf" \
    mcore_bridge==1.5.2 \
    megatron-core==0.18.0 \
    mathruler==0.1.0 \
    qwen_vl_utils==0.0.14 \
    datasets==5.0.0 \
    accelerate==1.14.0 \
    TransferQueue==0.1.8 \
    flash-linear-attention==0.5.1

RUN pip install --no-cache-dir cupy-rocm-7-0==14.1.1

FROM install_verl AS install_sglang

ENV ROCM_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel
ENV ROCM_HOME=${ROCM_PATH}
ENV HIP_PLATFORM=amd
ENV HIP_PATH=${ROCM_PATH}
ENV HIP_CLANG_PATH=${ROCM_PATH}/llvm/bin
ENV HIP_INCLUDE_PATH=${ROCM_PATH}/include
ENV HIP_LIB_PATH=${ROCM_PATH}/lib
ENV HIP_DEVICE_LIB_PATH=${ROCM_PATH}/lib/llvm/amdgcn/bitcode
ENV PATH="${ROCM_PATH}/bin:${ROCM_PATH}/llvm/bin:${PATH}"
ARG ROCM_CORE=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core
ENV LD_LIBRARY_PATH="${ROCM_CORE}/lib:${ROCM_CORE}/lib/rocm_sysdeps/lib:${ROCM_PATH}/lib:${ROCM_PATH}/lib/host-math/lib:${ROCM_PATH}/lib/rocm_sysdeps/lib:${LD_LIBRARY_PATH}"
ENV ROCPROFILER_LOG_LEVEL=fatal
ENV LIBRARY_PATH="${ROCM_PATH}/lib:${ROCM_PATH}/lib64"
ENV CPATH=${ROCM_PATH}/include
ENV PKG_CONFIG_PATH=${ROCM_PATH}/lib/pkgconfig
ENV GPU_ARCHS="gfx942;gfx950"

ENV SGLANG_USE_AITER=0
ENV SGLANG_ATTENTION_BACKEND=triton

ARG SGLANG_TAG="0801cc05ed202544e2e5c67d6b0a1d454de54fca"

RUN cd /workspace && git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && git checkout ${SGLANG_TAG} && \
    sed -zi 's/            out = torch.empty_like(x)\n            residual_out = torch.empty_like(x)\n            if post_residual_addition is not None:\n                residual = residual + post_residual_addition\n            fused_add_rms_norm(\n                out, x, residual_out, residual, self.weight.data, self.variance_epsilon\n            )\n            return out, residual_out/            if post_residual_addition is not None:\n                residual = residual + post_residual_addition\n            residual = residual.contiguous()\n            fused_add_rms_norm(x, residual, self.weight.data, self.variance_epsilon)\n            return x, residual/' \
      python/sglang/srt/layers/layernorm.py && \
    sed -zi 's/                out = torch.empty_like(x)\n                residual_out = torch.empty_like(x)\n                if post_residual_addition is not None:\n                    residual = residual + post_residual_addition\n                fused_add_rms_norm(\n                    out, x, residual_out, residual, w, self.variance_epsilon\n                )\n                return out, residual_out/                if post_residual_addition is not None:\n                    residual = residual + post_residual_addition\n                residual = residual.contiguous()\n                fused_add_rms_norm(x, residual, w, self.variance_epsilon)\n                return x, residual/' \
      python/sglang/srt/layers/layernorm.py && \
    sed -i 's|^\[\[tool.setuptools-rust.ext-modules\]\]|# [[tool.setuptools-rust.ext-modules]]|; s|^target = "sglang.srt.grpc._core"|# target = "sglang.srt.grpc._core"|; s|^path = "../rust/sglang-grpc/Cargo.toml"|# path = "../rust/sglang-grpc/Cargo.toml"|; s|^binding = "PyO3"|# binding = "PyO3"|' python/pyproject.toml && \
    pip install --no-cache-dir -e ./python --no-deps --no-build-isolation && \
    pip install --no-cache-dir IPython --upgrade-strategy only-if-needed --no-build-isolation && \
    pip install --no-cache-dir --no-deps --no-build-isolation soundfile==0.13.1 && \
    HIPCC_COMPILE_FLAGS_APPEND="--amdgpu-target=gfx90a;gfx942;gfx950 -D__HIP_PLATFORM_AMD__" \
    CFLAGS="-D__HIP_PLATFORM_AMD__ -I${ROCM_PATH}/include" \
    CXXFLAGS="-D__HIP_PLATFORM_AMD__ -I${ROCM_PATH}/include" \
    LDFLAGS="-L${ROCM_PATH}/lib" \
    pip install --no-cache-dir --no-deps --no-build-isolation "git+https://github.com/fzyzcjy/torch_memory_saver.git" && \
    cd sgl-kernel && cp pyproject_rocm.toml pyproject.toml && rm -rf build && \
    sed -i 's|f"--amdgpu-target={amdgpu_target}"|"--offload-arch=gfx942", "--offload-arch=gfx950"|' setup_rocm.py && \
    sed -i '/amdgpu_target not in \["gfx942", "gfx950"\]/,/sys.exit(1)/d' setup_rocm.py && \
    sed -zi 's/#if HIP_FP8_TYPE_FNUZ\n#include <c10\/util\/Float8_e4m3fnuz.h>\nusing FP8_TYPE = c10::Float8_e4m3fnuz;\nconstexpr auto FP8_E4M3_MAX = 224.0f;\n#else\n#if HIP_FP8_TYPE_E4M3\n#include <c10\/util\/Float8_e4m3fn.h>\nusing FP8_TYPE = c10::Float8_e4m3fn;\nC10_HOST_DEVICE constexpr auto FP8_E4M3_MAX = std::numeric_limits<FP8_TYPE>::max();\n#else\n#error "fp8 is not supported in this processor (arch < gfx942)."\n#endif  \/\/ HIP_FP8_TYPE_E4M3\n#endif  \/\/ HIP_FP8_TYPE_FNUZ/#if defined(__gfx942__)\n#include <c10\/util\/Float8_e4m3fnuz.h>\nusing FP8_TYPE = c10::Float8_e4m3fnuz;\nconstexpr auto FP8_E4M3_MAX = 224.0f;\n#else\n#include <c10\/util\/Float8_e4m3fn.h>\nusing FP8_TYPE = c10::Float8_e4m3fn;\nC10_HOST_DEVICE constexpr auto FP8_E4M3_MAX = std::numeric_limits<FP8_TYPE>::max();\n#endif  \/\/ __gfx942__/' include/utils.h && \
    python setup_rocm.py install && \
    rm -rf /root/.cache

ENV NVTE_FLASH_ATTN=1
ENV NVTE_FUSED_ATTN=0
ENV NVTE_USE_GROUPED_GEMM_TRITON=1
ENV NCCL_DEBUG=ERROR
ENV ROCPROFILER_REGISTER_ENABLED=0
ENV HSA_ENABLE_IPC_MODE_LEGACY=1
ENV PYTORCH_ALLOC_CONF=expandable_segments:True
ENV MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK=1
RUN SP="$(python -c 'import site; print(site.getsitepackages()[0])')"; \
    ln -sfn "$SP/_rocm_sdk_core/share/amd_smi/amdsmi" "$SP/amdsmi"
RUN apt-get update && apt-get install -y --no-install-recommends vim \
    && rm -rf /var/lib/apt/lists/*

# Clean build artifacts and caches
RUN rm -rf /*.whl \
    && rm -rf /root/.cache \
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
