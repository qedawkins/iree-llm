# Overview

This repository implements an example of a tokenizer module based on
SentencePiece for llama based llms. To build + run the samples here,
just initialize the submodules and build with clang

```
./init_submodules.sh

# ./build.sh
cmake -Bbuild -GNinja -S . \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++
```

To use the tokenizer, you need to generate a parameter archive for it. The
provided python script can do it alongside a runtime build with python bindings.
(Requires a HuggingFace token)

```
python gen_tokenizer_irpa.py
```

Releases and build instructions can be found here:

Releases page: https://github.com/openxla/iree/releases
Building from source: https://iree.dev/building-from-source/

To see an example of the tokenizer on its own, try:

```
./compile_and_run.sh
```

Note this requires a build of the iree-compiler too.

To try a full chat bot, separately download + compile a variant of llama
compatible with whatever tokenizer you generated through SHARK-Turbine. An
example can be found here:
https://github.com/nod-ai/SHARK-Turbine/tree/main/examples/llama2_inference

# License

Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
