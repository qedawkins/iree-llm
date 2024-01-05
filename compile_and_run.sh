~/iree-build/tools/iree-compile --iree-hal-target-backends=vulkan \
    --iree-vulkan-target-triple=rdna3-unknown-linux \
    src/tokenizer/test/example.mlir \
    -o module.vmfb
#./build/third_party/iree/tools/iree-run-module --device=local-sync \
~/iree-asan-build/tools/iree-run-module --device=vulkan \
    --module=build/src/tokenizer/module.so@create_tokenizer_module \
    --module=module.vmfb \
    --parameters=tokenizer.irpa \
    --trace_execution=true \
    --function=main
