~/iree-build/tools/iree-compile --iree-hal-target-backends=vmvx \
    chatbot/chatbot.mlir \
    -o module.vmfb
#./build/third_party/iree/tools/iree-run-module --device=local-sync \
~/iree-build/tools/iree-run-module --device=local-sync \
    --module=build/src/tokenizer/module.so@create_tokenizer_module \
    --module=module.vmfb \
    --parameters=tokenizer.irpa \
    --function=chat
    #--trace_execution=true \
