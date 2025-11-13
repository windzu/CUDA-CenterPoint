/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef COMMON_H_
#define COMMON_H_

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <math.h>

#include <algorithm>
#include <array>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

const unsigned int MAX_DET_NUM = 1000;  // nms_pre_max_size = 1000;
const unsigned int DET_CHANNEL = 11;
const unsigned int MAX_POINTS_NUM = 300000;
const unsigned int NUM_TASKS = 6;

#define checkCudaErrors(op)                                       \
  {                                                               \
    auto status = ((op));                                         \
    if (status != 0) {                                            \
      std::cout << "Cuda failure: " << cudaGetErrorString(status) \
                << " in file " << __FILE__ << ":" << __LINE__     \
                << " error status: " << status << std::endl;      \
      abort();                                                    \
    }                                                             \
  }

class Params {
 public:
  Params()
      : task_num_stride{0, 1, 3, 5, 6, 8},
        class_names{"car",         "truck",   "construction_vehicle",
                    "bus",         "trailer", "barrier",
                    "motorcycle",  "bicycle", "pedestrian",
                    "traffic_cone"} {}

  std::array<unsigned int, NUM_TASKS> task_num_stride;
  std::vector<std::string> class_names;

  float out_size_factor = 8.0F;
  std::array<float, 2> voxel_size{{0.075F, 0.075F}};
  std::array<float, 2> pc_range{{-54.0F, -54.0F}};
  float score_threshold = 0.1F;
  std::array<float, 6> post_center_range{
      {-61.2F, -61.2F, -10.0F, 61.2F, 61.2F, 10.0F}};
  float nms_iou_threshold = 0.2F;
  unsigned int nms_pre_max_size = MAX_DET_NUM;
  unsigned int nms_post_max_size = 83;

  float min_x_range = -54.0F;
  float max_x_range = 54.0F;
  float min_y_range = -54.0F;
  float max_y_range = 54.0F;
  float min_z_range = -5.0F;
  float max_z_range = 3.0F;
  float pillar_x_size = 0.075F;
  float pillar_y_size = 0.075F;
  float pillar_z_size = 0.2F;
  int max_points_per_voxel = 10;

  unsigned int max_voxels = 160000;
  unsigned int feature_num = 5;
  unsigned int max_points_num = MAX_POINTS_NUM;

  int getGridXSize() const {
    return static_cast<int>(
        std::round((max_x_range - min_x_range) / pillar_x_size));
  }

  int getGridYSize() const {
    return static_cast<int>(
        std::round((max_y_range - min_y_range) / pillar_y_size));
  }

  int getGridZSize() const {
    return static_cast<int>(
        std::round((max_z_range - min_z_range) / pillar_z_size));
  }
};

#endif