# Copyright 2024 The IREE Authors
# 
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(_NAME "iree_llm_dynamic_sentencepiece_module")
add_library(${_NAME} SHARED module.cc)
target_link_libraries(${_NAME}
  iree_base_base
  iree_hal_hal
  iree_modules_hal_types
  iree_vm_vm
  iree_vm_dynamic_api
  sentencepiece-static
)

# NOTE: this is only required because we want this sample to run on all
# platforms without needing to change the library name (libfoo.so/foo.dll).
set_target_properties(${_NAME}
  PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
    PREFIX ""
    OUTPUT_NAME "module"
)

target_compile_options(${_NAME} PRIVATE ${IREE_DEFAULT_COPTS})

add_subdirectory(test)
