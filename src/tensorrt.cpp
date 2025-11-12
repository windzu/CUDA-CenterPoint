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

#include "tensorrt.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "NvInfer.h"
#include "NvInferRuntime.h"

namespace TensorRT {

static class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR) {
      std::cerr << "[NVINFER LOG]: " << msg << std::endl;
    }
  }
} gLogger_;

template <typename T>
static void destroy_pointer(T* ptr) {
  if (ptr) delete ptr;
}

template <typename DimsT>
static std::string format_shape(const DimsT& shape) {
  std::ostringstream oss;
  for (int i = 0; i < shape.nbDims; ++i) {
    if (i) oss << " x ";
    oss << static_cast<long long>(shape.d[i]);
  }
  return oss.str();
}

static std::vector<uint8_t> load_file(const std::string& file) {
  std::ifstream in(file, std::ios::in | std::ios::binary);
  if (!in.is_open()) return {};

  in.seekg(0, std::ios::end);
  size_t length = in.tellg();

  std::vector<uint8_t> data;
  if (length > 0) {
    in.seekg(0, std::ios::beg);
    data.resize(length);
    in.read(reinterpret_cast<char*>(data.data()), length);
  }
  in.close();
  return data;
}

static const char* data_type_string(nvinfer1::DataType dt) {
  switch (dt) {
    case nvinfer1::DataType::kFLOAT:
      return "float32";
    case nvinfer1::DataType::kHALF:
      return "float16";
    case nvinfer1::DataType::kINT8:
      return "int8";
    case nvinfer1::DataType::kINT32:
      return "int32";
    case nvinfer1::DataType::kBOOL:
      return "bool";
    case nvinfer1::DataType::kUINT8:
      return "uint8";
#if NV_TENSORRT_MAJOR >= 10
    case nvinfer1::DataType::kINT64:
      return "int64";
    case nvinfer1::DataType::kFP8:
      return "fp8";
    case nvinfer1::DataType::kBF16:
      return "bf16";
    case nvinfer1::DataType::kINT4:
      return "int4";
#endif
    default:
      return "unknown";
  }
}

class EngineImpl : public Engine {
 public:
  bool load(const std::string& file) {
    auto data = load_file(file);
    if (data.empty()) {
      printf("Load engine %s failed.\n", file.c_str());
      return false;
    }
    return construct(data.data(), data.size(), file.c_str());
  }

  int64_t getBindingNumel(const std::string& name) override {
    auto iter = tensor_name_to_index_.find(name);
    if (iter == tensor_name_to_index_.end()) {
      printf("TensorRT can not find binding %s.\n", name.c_str());
      return 0;
    }
    auto dims = context_->getTensorShape(tensor_names_[iter->second].c_str());
    if (dims.nbDims <= 0) return 0;
    return std::accumulate(dims.d, dims.d + dims.nbDims, int64_t{1},
                           std::multiplies<int64_t>());
  }

  std::vector<int64_t> getBindingDims(const std::string& name) override {
    auto iter = tensor_name_to_index_.find(name);
    if (iter == tensor_name_to_index_.end()) {
      printf("TensorRT can not find binding %s.\n", name.c_str());
      return {};
    }
    auto dims = context_->getTensorShape(tensor_names_[iter->second].c_str());
    std::vector<int64_t> output(dims.nbDims);
    for (int i = 0; i < dims.nbDims; ++i) {
      output[i] = static_cast<int64_t>(dims.d[i]);
    }
    return output;
  }

  bool forward(const std::initializer_list<void*>& buffers,
               void* stream = nullptr) override {
    if (!context_) {
      printf("TensorRT context is null.\n");
      return false;
    }

    std::vector<void*> buffer_list(buffers.begin(), buffers.end());
    const int expected = engine_->getNbIOTensors();
    if (static_cast<int>(buffer_list.size()) != expected) {
      printf("TensorRT expects %d bindings but %zu provided.\n", expected,
             buffer_list.size());
      return false;
    }

    for (int i = 0; i < expected; ++i) {
      const char* tensor_name = tensor_names_[i].c_str();
      if (!context_->setTensorAddress(tensor_name, buffer_list[i])) {
        printf("Failed to set tensor address for %s.\n", tensor_name);
        return false;
      }
    }

    return context_->enqueueV3(reinterpret_cast<cudaStream_t>(stream));
  }

  void print() override {
    if (!context_) {
      printf("Infer print, nullptr.\n");
      return;
    }

    std::vector<int> input_indices;
    std::vector<int> output_indices;
    const int total = engine_->getNbIOTensors();
    input_indices.reserve(total);
    output_indices.reserve(total);

    for (int i = 0; i < total; ++i) {
      const auto mode = engine_->getTensorIOMode(tensor_names_[i].c_str());
      if (mode == nvinfer1::TensorIOMode::kINPUT)
        input_indices.push_back(i);
      else
        output_indices.push_back(i);
    }

    printf("Engine %p detail\n", this);
    printf("Inputs: %zu\n", input_indices.size());
    for (size_t idx = 0; idx < input_indices.size(); ++idx) {
      int binding = input_indices[idx];
      const char* name = tensor_names_[binding].c_str();
      auto dims = context_->getTensorShape(name);
      auto dtype = engine_->getTensorDataType(name);
      printf("\t%zu.%s : \tshape {%s}, %s\n", idx, name,
             format_shape(dims).c_str(), data_type_string(dtype));
    }

    printf("Outputs: %zu\n", output_indices.size());
    for (size_t idx = 0; idx < output_indices.size(); ++idx) {
      int binding = output_indices[idx];
      const char* name = tensor_names_[binding].c_str();
      auto dims = context_->getTensorShape(name);
      auto dtype = engine_->getTensorDataType(name);
      printf("\t%zu.%s : \tshape {%s}, %s\n", idx, name,
             format_shape(dims).c_str(), data_type_string(dtype));
    }
  }

 private:
  bool construct(const void* data, size_t size, const char* message_name) {
    runtime_ = std::shared_ptr<nvinfer1::IRuntime>(
        nvinfer1::createInferRuntime(gLogger_),
        destroy_pointer<nvinfer1::IRuntime>);
    if (!runtime_) {
      printf("Failed to create TensorRT runtime: %s.\n", message_name);
      return false;
    }

    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime_->deserializeCudaEngine(data, size),
        destroy_pointer<nvinfer1::ICudaEngine>);
    if (!engine_) {
      printf("Failed to deserialize engine: %s.\n", message_name);
      return false;
    }

    context_ = std::shared_ptr<nvinfer1::IExecutionContext>(
        engine_->createExecutionContext(),
        destroy_pointer<nvinfer1::IExecutionContext>);
    if (!context_) {
      printf("Failed to create execution context: %s.\n", message_name);
      return false;
    }

    tensor_names_.clear();
    tensor_name_to_index_.clear();
    const int nb = engine_->getNbIOTensors();
    tensor_names_.reserve(nb);
    for (int i = 0; i < nb; ++i) {
      const char* tensor_name = engine_->getIOTensorName(i);
      tensor_names_.emplace_back(tensor_name ? tensor_name : "");
      tensor_name_to_index_[tensor_names_.back()] = i;
    }

    for (const auto& name : tensor_names_) {
      if (engine_->getTensorIOMode(name.c_str()) !=
          nvinfer1::TensorIOMode::kINPUT)
        continue;
      auto dims = engine_->getTensorShape(name.c_str());
      bool has_dynamic_dim = false;
      for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] == -1) {
          has_dynamic_dim = true;
          break;
        }
      }
      if (!has_dynamic_dim) {
        if (!context_->setInputShape(name.c_str(), dims)) {
          printf("Failed to set static input shape for %s.\n", name.c_str());
          return false;
        }
      }
    }
    return true;
  }

 private:
  std::shared_ptr<nvinfer1::IRuntime> runtime_;
  std::shared_ptr<nvinfer1::ICudaEngine> engine_;
  std::shared_ptr<nvinfer1::IExecutionContext> context_;
  std::vector<std::string> tensor_names_;
  std::unordered_map<std::string, int> tensor_name_to_index_;
};

std::shared_ptr<Engine> load(const std::string& file) {
  std::shared_ptr<EngineImpl> impl(new EngineImpl());
  if (!impl->load(file)) impl.reset();
  return impl;
}

}  // namespace TensorRT