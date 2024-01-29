# Copyright 2024 The IREE Authors
# 
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.21...3.24)

#-------------------------------------------------------------------------------
# Project configuration
#-------------------------------------------------------------------------------

project(iree-llm C CXX)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#-------------------------------------------------------------------------------
# Third party dependencies
#-------------------------------------------------------------------------------

# Extend module path to find MLIR CMake modules.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/mlir")

# Disable core project features, we only need pieces of the runtime.
set(IREE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_COMPILER OFF CACHE BOOL "" FORCE)
#set(IREE_ENABLE_ASAN ON CACHE BOOL "" FORCE)

# # Dynamic modules currently don't support source file locations.
# # "-DIREE_STATUS_MODE=0"
# set(IREE_DEFAULT_COPTS
#     "${IREE_DEFAULT_COPTS}"
#     "-DIREE_STATUS_FEATURES=2"
# )


add_subdirectory(third_party/iree EXCLUDE_FROM_ALL)
add_subdirectory(third_party/sentencepiece EXCLUDE_FROM_ALL)
include_directories(third_party/sentencepiece/src)

#-------------------------------------------------------------------------------
# Individual samples
#-------------------------------------------------------------------------------

add_subdirectory(src/tokenizer)
