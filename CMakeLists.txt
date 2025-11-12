# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

cmake_minimum_required(VERSION 2.8.7)
project(centerpoint)
set(arch ${CMAKE_HOST_SYSTEM_PROCESSOR})

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS_RELEASE "-Wextra -Wall -Wno-deprecated-declarations -O3")

if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  message("Using Debug Mode")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -G --ptxas-options=-v)
endif()

if(NOT DEFINED ENV{SPCONV_CUDA_VERSION} OR "$ENV{SPCONV_CUDA_VERSION}" STREQUAL "")
  set(ENV{SPCONV_CUDA_VERSION} "12.8")
endif()

# --- Find CUDA , CUDNN , TensorRT Modules ---
# 1. 自定义 FindCUDNN.cmake FindTENSORRT.cmake 模块路径
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/Modules")

# 2. Find CUDA , CUDNN , TensorRT
find_package(CUDA)
find_package(CUDAToolkit)
find_package(CUDNN REQUIRED)
find_package(TENSORRT REQUIRED)

# 3. Echo Status Info
message(STATUS "== CMake probe ==")
message(STATUS "CUDAToolkit_FOUND = ${CUDAToolkit_FOUND}")
message(STATUS "CUDA_INCLUDE_DIRS = ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDNN_FOUND       = ${CUDNN_FOUND}")
message(STATUS "CUDNN_LIBRARY     = ${CUDNN_LIBRARY}")
message(STATUS "TENSORRT_FOUND    = ${TENSORRT_FOUND}")
message(STATUS "TENSORRT_VERSION_STRING  = ${TENSORRT_VERSION_STRING}")
message(STATUS "TENSORRT_INCLUDES = ${TENSORRT_INCLUDE_DIRS}")
message(STATUS "TENSORRT_LIBS     = ${TENSORRT_LIBRARIES}")

if(NOT (CUDAToolkit_FOUND AND CUDNN_FOUND AND TENSORRT_FOUND))
  message(WARNING "cuda, cudnn, tensorrt libraries are not found")
  return()
endif()

if(TENSORRT_VERSION_STRING VERSION_LESS 8.5)
  message(WARNING "Unsupported version TensorRT ${TENSORRT_VERSION_STRING} detected. This package requires TensorRT 8.5 or later.")
  return()
endif()
# -----------------------------------------

set(SPCONV_ROOT "${CMAKE_CURRENT_LIST_DIR}/third_party/spconv")
set(SPCONV_INCLUDE "${SPCONV_ROOT}/include")
set(SPCONV_LIB_DIR "${SPCONV_ROOT}/lib/${arch}_cuda$ENV{SPCONV_CUDA_VERSION}")
set(SPCONV_PARSER_DIR "${SPCONV_ROOT}/parser")
set(SPCONV_ONNX_DIR "${SPCONV_ROOT}/onnx")

find_library(SPCONV_LIBRARY spconv PATHS "${SPCONV_LIB_DIR}" NO_DEFAULT_PATH)
if(NOT SPCONV_LIBRARY)
  message(FATAL_ERROR "Unable to resolve spconv library inside ${SPCONV_LIB_DIR}.")
endif()

find_package(Protobuf REQUIRED)

set(SPCONV_PROTO_GEN_DIR "${CMAKE_BINARY_DIR}/generated/spconv")
set(SPCONV_PROTO_OUTPUT_DIR "${SPCONV_PROTO_GEN_DIR}/onnx")
set(SPCONV_PROTO_FILES
  "${SPCONV_ONNX_DIR}/onnx-ml.proto"
  "${SPCONV_ONNX_DIR}/onnx-operators-ml.proto"
)

set(SPCONV_PROTO_SRCS
  "${SPCONV_PROTO_OUTPUT_DIR}/onnx-ml.pb.cc"
  "${SPCONV_PROTO_OUTPUT_DIR}/onnx-operators-ml.pb.cc"
)
set(SPCONV_PROTO_HDRS
  "${SPCONV_PROTO_OUTPUT_DIR}/onnx-ml.pb.h"
  "${SPCONV_PROTO_OUTPUT_DIR}/onnx-operators-ml.pb.h"
)

if(NOT Protobuf_PROTOC_EXECUTABLE)
  message(FATAL_ERROR "Failed to locate protoc executable via find_package(Protobuf).")
endif()

add_custom_command(
  OUTPUT ${SPCONV_PROTO_SRCS} ${SPCONV_PROTO_HDRS}
  COMMAND ${CMAKE_COMMAND} -E make_directory "${SPCONV_PROTO_OUTPUT_DIR}"
  COMMAND ${CMAKE_COMMAND} -E env PROTO_OUT_DIR=${SPCONV_PROTO_OUTPUT_DIR} PROTOC_BIN=${Protobuf_PROTOC_EXECUTABLE} bash "${SPCONV_ONNX_DIR}/make_pb.sh"
  DEPENDS ${SPCONV_PROTO_FILES} "${SPCONV_ONNX_DIR}/make_pb.sh"
  WORKING_DIRECTORY "${SPCONV_ONNX_DIR}"
  COMMENT "Generating spconv ONNX protobuf sources"
  VERBATIM
)

add_custom_target(spconv_proto_generated DEPENDS ${SPCONV_PROTO_SRCS} ${SPCONV_PROTO_HDRS})



include_directories(
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIRS}
  ${SPCONV_INCLUDE}
  ${SPCONV_PARSER_DIR}
  ${SPCONV_PROTO_OUTPUT_DIR}
  ${Protobuf_INCLUDE_DIRS}
  include
)

file(GLOB_RECURSE SOURCE_FILES 
  src/*.cu 
  src/*.cpp
)

list(APPEND SOURCE_FILES
  "${SPCONV_PARSER_DIR}/onnx-parser.cpp"
  ${SPCONV_PROTO_SRCS}
)

cuda_add_executable(${PROJECT_NAME} main.cpp ${SOURCE_FILES})
add_dependencies(${PROJECT_NAME} spconv_proto_generated)

target_link_libraries(${PROJECT_NAME}
  ${TENSORRT_LIBRARIES}
  ${SPCONV_LIBRARY}
  ${Protobuf_LIBRARIES}
)