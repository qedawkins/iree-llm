~/iree-build/tools/iree-compile --iree-hal-target-backends=vulkan \
    --iree-vulkan-target-triple=rdna3-unknown-linux \
    chatbot/chatbot.mlir \
    -o module.vmfb
#./build/third_party/iree/tools/iree-run-module --device=local-sync \
~/iree-asan-build/tools/iree-run-module --device=vulkan \
    --module=build/src/tokenizer/module.so@create_tokenizer_module \
    --module=/tmp/llama.vmfb \
    --module=module.vmfb \
    --parameters=tokenizer.irpa \
    --parameters=model=/home/quinn/model_testing/turbine/Llama2_7b_i4quant.irpa \
    --function=chat
    #--trace_execution=true \
    #--parameters=model=/home/quinn/model_testing/turbine/llama_f16.safetensors \
