#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

set -euo pipefail

project_folder=$(realpath "$(dirname "${BASH_SOURCE[-1]}")")
cd "${project_folder}"

protoc_bin=${PROTOC_BIN:-/usr/bin/protoc}
out_dir=${PROTO_OUT_DIR:-${project_folder}}

mkdir -p "${out_dir}"

"${protoc_bin}" onnx-ml.proto --cpp_out="${out_dir}"
"${protoc_bin}" onnx-operators-ml.proto --cpp_out="${out_dir}"

echo "Generated protobuf sources in ${out_dir}"
