# - Find cuDNN
#
# Finds the NVIDIA cuDNN library.
# This module is written specifically based on user-provided paths.
# [V2: Added ARM (aarch64) compatibility]
#
# Result Variables:
#
#  CUDNN_FOUND:           True if cuDNN was found.
#  CUDNN_INCLUDE_DIRS:  The cuDNN include directories.
#  CUDNN_LIBRARIES:     The cuDNN libraries.
#  CUDNN_VERSION_STRING: The cuDNN version (e.g., "8.9.5").
#
# Targets:
#
#  CUDNN::cudnn:        An IMPORTED target for cuDNN.
#

include(FindPackageHandleStandardArgs)

# --- 新增：设置架构特定路径 ---
# /usr/include 是架构无关的
set(CUDNN_ARCH_INCLUDE_HINT /usr/include)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    # 适用于 ARM64 架构，例如 NVIDIA Jetson Orin
    set(CUDNN_ARCH_LIB_HINT /usr/lib/aarch64-linux-gnu)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    # 适用于 x86_64 架构 (基于用户的原始路径)
    set(CUDNN_ARCH_LIB_HINT /usr/lib/x86_64-linux-gnu)
else()
    # 其他架构的通用回退
    set(CUDNN_ARCH_LIB_HINT /usr/lib)
endif()

# --- 1. Find Include Directory ---
find_path(CUDNN_INCLUDE_DIR
          NAMES cudnn_version.h
          HINTS ${CUDNN_ARCH_INCLUDE_HINT}  # <-- 使用变量
          PATHS /usr/local/cuda/include
          NO_DEFAULT_PATH
)
if(NOT CUDNN_INCLUDE_DIR)
    find_path(CUDNN_INCLUDE_DIR NAMES cudnn_version.h)
endif()

# --- 2. Find Library ---
find_library(CUDNN_LIBRARY
             NAMES cudnn
             HINTS ${CUDNN_ARCH_LIB_HINT}  # <-- 使用变量
             PATHS /usr/local/cuda/lib64
             NO_DEFAULT_PATH
)
if(NOT CUDNN_LIBRARY)
    find_library(CUDNN_LIBRARY NAMES cudnn)
endif()


# --- 3. Extract Version ---
# (此部分保持不变)
if(CUDNN_INCLUDE_DIR AND EXISTS "${CUDNN_INCLUDE_DIR}/cudnn_version.h")
    file(STRINGS "${CUDNN_INCLUDE_DIR}/cudnn_version.h" CUDNN_VERSION_MAJOR_LINE REGEX "^#define CUDNN_MAJOR")
    file(STRINGS "${CUDNN_INCLUDE_DIR}/cudnn_version.h" CUDNN_VERSION_MINOR_LINE REGEX "^#define CUDNN_MINOR")
    file(STRINGS "${CUDNN_INCLUDE_DIR}/cudnn_version.h" CUDNN_VERSION_PATCH_LINE REGEX "^#define CUDNN_PATCHLEVEL")

    string(REGEX REPLACE ".*CUDNN_MAJOR +([0-9]+).*" "\\1" CUDNN_VERSION_MAJOR "${CUDNN_VERSION_MAJOR_LINE}")
    string(REGEX REPLACE ".*CUDNN_MINOR +([0-9]+).*" "\\1" CUDNN_VERSION_MINOR "${CUDNN_VERSION_MINOR_LINE}")
    string(REGEX REPLACE ".*CUDNN_PATCHLEVEL +([0-9]+).*" "\\1" CUDNN_VERSION_PATCH "${CUDNN_VERSION_PATCH_LINE}")

    set(CUDNN_VERSION_STRING "${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCH}")
endif()

# --- 4. Handle Standard Arguments ---
# (此部分保持不变)
find_package_handle_standard_args(CUDNN
                                  FOUND_VAR CUDNN_FOUND
                                  REQUIRED_VARS CUDNN_LIBRARY CUDNN_INCLUDE_DIR
                                  VERSION_VAR CUDNN_VERSION_STRING)

# --- 5. Create Target ---
# (此部分保持不变)
if(CUDNN_FOUND)
    set(CUDNN_LIBRARIES ${CUDNN_LIBRARY})
    set(CUDNN_INCLUDE_DIRS ${CUDNN_INCLUDE_DIR})

    if(NOT TARGET CUDNN::cudnn)
        add_library(CUDNN::cudnn UNKNOWN IMPORTED)
        set_target_properties(CUDNN::cudnn PROPERTIES
                              IMPORTED_LOCATION "${CUDNN_LIBRARY}"
                              INTERFACE_INCLUDE_DIRECTORIES "${CUDNN_INCLUDE_DIR}")
    endif()
endif()

mark_as_advanced(CUDNN_INCLUDE_DIR CUDNN_LIBRARY)