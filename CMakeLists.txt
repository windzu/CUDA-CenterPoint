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

cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0023 NEW)
cmake_policy(SET CMP0104 NEW)

project(centerpoint LANGUAGES C CXX CUDA)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS_RELEASE "-Wextra -Wall -Wno-deprecated-declarations -O3")
option(CENTERPOINT_ENABLE_INSTALL "Enable install rules for centerpoint" ON)

set(arch ${CMAKE_HOST_SYSTEM_PROCESSOR})

# --- Set spconv CUDA Version ---
if(NOT DEFINED SPCONV_CUDA_VERSION)
  if(DEFINED ENV{SPCONV_CUDA_VERSION})
    set(SPCONV_CUDA_VERSION "$ENV{SPCONV_CUDA_VERSION}" CACHE STRING "spconv CUDA version")
  else()
    set(SPCONV_CUDA_VERSION "12.8" CACHE STRING "spconv CUDA version")
  endif()
endif()
message(STATUS "Using SPCONV_CUDA_VERSION: ${SPCONV_CUDA_VERSION}")

# --- Find CUDA , CUDNN , TensorRT Modules ---
# 1. 自定义 FindCUDNN.cmake FindTENSORRT.cmake 模块路径
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/Modules")

# 2. Find CUDA , CUDNN , TensorRT
# find_package(CUDA)
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
set(SPCONV_LIB_DIR "${SPCONV_ROOT}/lib/${arch}_cuda${SPCONV_CUDA_VERSION}")
set(SPCONV_PARSER_DIR "${SPCONV_ROOT}/parser")
set(SPCONV_ONNX_DIR "${SPCONV_ROOT}/onnx")

find_library(SPCONV_LIBRARY spconv PATHS "${SPCONV_LIB_DIR}" NO_DEFAULT_PATH)
if(NOT SPCONV_LIBRARY)
  message(FATAL_ERROR "Unable to resolve spconv library inside ${SPCONV_LIB_DIR}.")
endif()

find_package(Protobuf REQUIRED)

set(SPCONV_PROTO_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/spconv")
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


file(GLOB_RECURSE SOURCE_FILES 
  src/*.cu 
  src/*.cpp
)

list(APPEND SOURCE_FILES
  "${SPCONV_PARSER_DIR}/onnx-parser.cpp"
  ${SPCONV_PROTO_SRCS}
)

# Build Core Library
add_library(${PROJECT_NAME}_core ${SOURCE_FILES})
add_dependencies(${PROJECT_NAME}_core spconv_proto_generated)

set_target_properties(${PROJECT_NAME}_core PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
)

target_include_directories(${PROJECT_NAME}_core
  PUBLIC
    # self header files
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    
    # all dependency headers
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIRS}
    ${SPCONV_INCLUDE}
    ${SPCONV_PARSER_DIR}
    ${SPCONV_PROTO_OUTPUT_DIR}
    ${Protobuf_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}_core
  PUBLIC
    ${TENSORRT_LIBRARIES}
    ${SPCONV_LIBRARY}
    ${Protobuf_LIBRARIES}
)

# Build Executable
option(CENTERPOINT_BUILD_EXECUTABLE "Build the standalone centerpoint executable" OFF)
if(CENTERPOINT_BUILD_EXECUTABLE)
  # (保持你原来的逻辑)
  if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/main.cpp")
    add_executable(${PROJECT_NAME} main.cpp)
    add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_core)

    target_link_libraries(${PROJECT_NAME}
      ${PROJECT_NAME}_core
    )
  endif()
endif()

# Install
if(CENTERPOINT_ENABLE_INSTALL)
  install(TARGETS ${PROJECT_NAME}_core
    EXPORT centerpointTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin)

  # 安装 spconv lib and include
  set(SPCONV_LIB_DIR "${SPCONV_ROOT}/lib/${arch}_cuda${SPCONV_CUDA_VERSION}")
  install(FILES "${SPCONV_LIB_DIR}/libspconv.so" DESTINATION lib)
  install(DIRECTORY "${SPCONV_INCLUDE}/" DESTINATION include)

  install(DIRECTORY include/ DESTINATION include)
endif()