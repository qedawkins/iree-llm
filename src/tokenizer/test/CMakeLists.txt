# Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.

iree_lit_test_suite(
  NAME
    lit
  SRCS
    "example.mlir"
  TOOLS
    FileCheck
    iree-compile
    iree-run-module
    iree_samples_custom_module_dynamic_module
  LABELS
    "driver=local-sync"
    "hostonly"
)
