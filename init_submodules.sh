#!/bin/bash

git submodule update --init
cd third_party/iree
git submodule update --init third_party/benchmark third_party/cpuinfo third_party/flatcc third_party/googletest third_party/libyaml third_party/musl third_party/spirv_cross third_party/tracy third_party/vulkan_headers third_party/webgpu-headers
cd ../../
