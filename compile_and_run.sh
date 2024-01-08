iree-compile --iree-hal-target-backends=vulkan \
    --iree-vulkan-target-triple=rdna3-unknown-linux \
    src/tokenizer/test/example.mlir \
    -o module.vmfb

iree-run-module --device=vulkan \
    --module=build/src/tokenizer/module.so@create_tokenizer_module \
    --module=module.vmfb \
    --parameters=tokenizer.irpa \
    --trace_execution=true \
    --function=main
