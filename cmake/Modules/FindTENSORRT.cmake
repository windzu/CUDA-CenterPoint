# - Find TensorRT
#
# Finds the NVIDIA TensorRT library.
# This module is written specifically based on user-provided paths.
# [V2: Added ARM (aarch64) compatibility]
#
# Result Variables:
#
#  TENSORRT_FOUND:           True if TensorRT was found.
#  TENSORRT_INCLUDE_DIRS:  The TensorRT include directories.
#  TENSORRT_LIBRARIES:     The TensorRT libraries (nvinfer, nvinfer_plugin, etc.).
#  TENSORRT_VERSION_STRING: The TensorRT version (e.g., "8.6.1").
#
# Targets:
#
#  TENSORRT::tensorrt:    An INTERFACE target for TensorRT.
#

include(FindPackageHandleStandardArgs)

# --- 新增：设置架构特定路径 ---
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    # 适用于 ARM64 架构，例如 NVIDIA Jetson Orin
    set(TENSORRT_ARCH_INCLUDE_HINT /usr/include/aarch64-linux-gnu)
    set(TENSORRT_ARCH_LIB_HINT /usr/lib/aarch64-linux-gnu)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    # 适用于 x86_64 架构 (基于用户的原始路径)
    set(TENSORRT_ARCH_INCLUDE_HINT /usr/include/x86_64-linux-gnu)
    set(TENSORRT_ARCH_LIB_HINT /usr/lib/x86_64-linux-gnu)
else()
    # 其他架构的通用回退
    set(TENSORRT_ARCH_INCLUDE_HINT /usr/include)
    set(TENSORRT_ARCH_LIB_HINT /usr/lib)
endif()

# --- 1. Find Include Directory ---
find_path(TENSORRT_INCLUDE_DIR
          NAMES NvInfer.h
          HINTS ${TENSORRT_ARCH_INCLUDE_HINT}  # <-- 使用变量
          PATHS /usr/src/tensorrt/include /opt/tensorrt/include
          NO_DEFAULT_PATH
)
if(NOT TENSORRT_INCLUDE_DIR)
    find_path(TENSORRT_INCLUDE_DIR NAMES NvInfer.h)
endif()


# --- 2. Find Libraries ---
set(TENSORRT_COMMON_LIB_PATH ${TENSORRT_ARCH_LIB_HINT}) # <-- 使用变量
set(TENSORRT_COMMON_LIB_FALLBACK_PATHS /usr/src/tensorrt/lib /opt/tensorrt/lib)

# Find main library
find_library(TENSORRT_LIBRARY
             NAMES nvinfer
             HINTS ${TENSORRT_COMMON_LIB_PATH}
             PATHS ${TENSORRT_COMMON_LIB_FALLBACK_PATHS}
             NO_DEFAULT_PATH
)
if(NOT TENSORRT_LIBRARY)
    find_library(TENSORRT_LIBRARY NAMES nvinfer)
endif()

# Find other common TRT libraries
find_library(TENSORRT_PLUGIN_LIBRARY
             NAMES nvinfer_plugin
             HINTS ${TENSORRT_COMMON_LIB_PATH}
             PATHS ${TENSORRT_COMMON_LIB_FALLBACK_PATHS}
             NO_DEFAULT_PATH
)
find_library(TENSORRT_PARSERS_LIBRARY
             NAMES nvparsers
             HINTS ${TENSORRT_COMMON_LIB_PATH}
             PATHS ${TENSORRT_COMMON_LIB_FALLBACK_PATHS}
             NO_DEFAULT_PATH
)
find_library(TENSORRT_ONNX_PARSER_LIBRARY
             NAMES nvonnxparser
             HINTS ${TENSORRT_COMMON_LIB_PATH}
             PATHS ${TENSORRT_COMMON_LIB_FALLBACK_PATHS}
             NO_DEFAULT_PATH
)


# --- 3. Extract Version ---
# (此部分保持不变)
if(TENSORRT_INCLUDE_DIR AND EXISTS "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h")
    file(STRINGS "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h" TENSORRT_VERSION_MAJOR_LINE REGEX "^#define NV_TENSORRT_MAJOR")
    file(STRINGS "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h" TENSORRT_VERSION_MINOR_LINE REGEX "^#define NV_TENSORRT_MINOR")
    file(STRINGS "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h" TENSORRT_VERSION_PATCH_LINE REGEX "^#define NV_TENSORRT_PATCH")

    string(REGEX REPLACE ".*NV_TENSORRT_MAJOR +([0-9]+).*" "\\1" TENSORRT_VERSION_MAJOR "${TENSORRT_VERSION_MAJOR_LINE}")
    string(REGEX REPLACE ".*NV_TENSORRT_MINOR +([0-9]+).*" "\\1" TENSORRT_VERSION_MINOR "${TENSORRT_VERSION_MINOR_LINE}")
    string(REGEX REPLACE ".*NV_TENSORRT_PATCH +([0-9]+).*" "\\1" TENSORRT_VERSION_PATCH "${TENSORRT_VERSION_PATCH_LINE}")

    set(TENSORRT_VERSION_STRING "${TENSORRT_VERSION_MAJOR}.${TENSORRT_VERSION_MINOR}.${TENSORRT_VERSION_PATCH}")
endif()

# --- 4. Handle Standard Arguments ---
# (此部分保持不变)
find_package_handle_standard_args(TENSORRT
                                  FOUND_VAR TENSORRT_FOUND
                                  REQUIRED_VARS TENSORRT_LIBRARY TENSORRT_INCLUDE_DIR
                                  VERSION_VAR TENSORRT_VERSION_STRING)

# --- 5. Create Target ---
# (此部分保持不变)
if(TENSORRT_FOUND)
    # Combine all found libraries into one list
    set(TENSORRT_LIBRARIES ${TENSORRT_LIBRARY})
    
    if(TENSORRT_PLUGIN_LIBRARY)
        list(APPEND TENSORRT_LIBRARIES ${TENSORRT_PLUGIN_LIBRARY})
    endif()
    if(TENSORRT_PARSERS_LIBRARY)
        list(APPEND TENSORRT_LIBRARIES ${TENSORRT_PARSERS_LIBRARY})
    endif()
    if(TENSORRT_ONNX_PARSER_LIBRARY)
        list(APPEND TENSORRT_LIBRARIES ${TENSORRT_ONNX_PARSER_LIBRARY})
    endif()
    
    set(TENSORRT_INCLUDE_DIRS ${TENSORRT_INCLUDE_DIR})

    if(NOT TARGET TENSORRT::tensorrt)
        # Create an INTERFACE target. This is easier for consumers.
        add_library(TENSORRT::tensorrt INTERFACE IMPORTED)
        set_target_properties(TENSORRT::tensorrt PROPERTIES
                              INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_INCLUDE_DIRS}"
                              INTERFACE_LINK_LIBRARIES "${TENSORRT_LIBRARIES}")
    endif()
endif()

mark_as_advanced(TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY TENSORRT_PLUGIN_LIBRARY TENSORRT_PARSERS_LIBRARY TENSORRT_ONNX_PARSER_LIBRARY)