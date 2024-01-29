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

Note: This repository is being developed for inclusion in the iree project
in some form.

Copyright 2024 The IREE Authors

Licensed under the Apache License v2.0 with LLVM Exceptions.
See https://llvm.org/LICENSE.txt for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
