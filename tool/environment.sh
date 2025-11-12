#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

export TensorRT_Lib=${TensorRT_Lib:-/usr/lib/x86_64-linux-gnu}
export TensorRT_Inc=${TensorRT_Inc:-/usr/include/x86_64-linux-gnu}
export TensorRT_Bin=${TensorRT_Bin:-/opt/tensorrt/bin}

export CUDA_Lib=${CUDA_Lib:-/usr/local/cuda/lib64}
export CUDA_Inc=${CUDA_Inc:-/usr/local/cuda/include}
export CUDA_Bin=${CUDA_Bin:-/usr/local/cuda/bin}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}

export CUDNN_Lib=${CUDNN_Lib:-/usr/lib/x86_64-linux-gnu}

export SPCONV_CUDA_VERSION=${SPCONV_CUDA_VERSION:-12.8}

export ConfigurationStatus=Failed
if [ ! -f "${TensorRT_Bin}/trtexec" ]; then
    echo "Cannot locate ${TensorRT_Bin}/trtexec. Please ensure TensorRT is installed correctly."
    return
fi

if [ ! -f "${CUDA_Bin}/nvcc" ]; then
    echo "Cannot locate ${CUDA_Bin}/nvcc. Please ensure CUDA is installed correctly."
    return
fi

export PATH=${TensorRT_Bin}:${CUDA_Bin}:$PATH
export LD_LIBRARY_PATH=${TensorRT_Lib}:${CUDA_Lib}:${CUDNN_Lib}:$LD_LIBRARY_PATH

export ConfigurationStatus=Success

echo "CenterPoint environment configured."
