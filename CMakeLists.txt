# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.21...3.23)

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

add_subdirectory(third_party/iree)
add_subdirectory(third_party/sentencepiece)

#-------------------------------------------------------------------------------
# Individual samples
#-------------------------------------------------------------------------------

add_subdirectory(src/tokenizer)
