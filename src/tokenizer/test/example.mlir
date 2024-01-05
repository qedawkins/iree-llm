// RUN: iree-compile %s --iree-hal-target-backends=vmvx | \
// RUN: iree-run-module \
// RUN:     --device=local-sync \
// RUN:     --module=$IREE_BINARY_DIR/samples/custom_module/dynamic/module$IREE_DYLIB_EXT@create_custom_module \
// RUN:     --module=- \
// RUN:     --function=main | \
// RUN: FileCheck %s

module @example {

  util.global private @serialized_tokenizer = #stream.parameter.named<"tokenizer"> : tensor<499723xi8>

  func.func private @print_buffer(%stream0: !io_stream.handle, %buf: !util.buffer) {
    %c0_2 = arith.constant 0 : index
    %buf_size = util.buffer.size %buf : !util.buffer
    io_stream.write.bytes(%stream0, %buf, %c0_2, %buf_size) : (!io_stream.handle, !util.buffer, index, index) -> ()
    %newline0 = arith.constant 10 : i8  // \n
    io_stream.write.byte(%stream0, %newline0) : (!io_stream.handle, i8) -> ()
    return
  }


  //===--------------------------------------------------------------------===//
  // Imports
  //===--------------------------------------------------------------------===//
  // Creates a new tokenizer (on the host) with contents from the given tensor.
  // Probably don't use this.
  func.func private @tokenizer.load_spm.from_tensor(tensor<?xi8>) -> !tokenizer.spm

  // Creates a new tokenizer with the contents of the given buffer.
  func.func private @tokenizer.load_spm.from_buffer(!util.buffer) -> !tokenizer.spm

  // Returns an array of 64-bit tokens based on the tokenizer and given text.
  func.func private @tokenizer.encode_i64(!tokenizer.spm, !util.buffer) -> !util.buffer

  //===--------------------------------------------------------------------===//
  // Sample methods
  //===--------------------------------------------------------------------===//

  // CHECK-LABEL: EXEC @main
  func.func @main() -> tensor<?xi64> {
    %stdout = io_stream.console.stdout : !io_stream.handle
    %text = util.buffer.constant : !util.buffer = "please tokenize me :D"
    func.call @print_buffer(%stdout, %text) : (!io_stream.handle, !util.buffer) -> ()

    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %serialized_tokenizer = util.global.load @serialized_tokenizer : tensor<499723xi8>
    %cast = tensor.cast %serialized_tokenizer : tensor<499723xi8> to tensor<?xi8>
    %tokenizer = call @tokenizer.load_spm.from_tensor(%cast) : (tensor<?xi8>) -> !tokenizer.spm

    %tokens = call @tokenizer.encode_i64(%tokenizer, %text) : (!tokenizer.spm, !util.buffer) -> !util.buffer

    // We need to get the device allocator to import the util buffer.
    %device_0 = hal.devices.get %c0 : !hal.device
    %allocator = hal.device.allocator<%device_0 : !hal.device> : !hal.allocator

    %affinity = arith.constant -1 : i64
    %tok_bytes = util.buffer.size %tokens : !util.buffer
    %ok, %ref = hal.allocator.import<%allocator : !hal.allocator>
            source(%tokens : !util.buffer)[%c0, %tok_bytes]
            affinity(%affinity) type(DeviceLocal) usage("TransferSource|TransferTarget|Transfer|DispatchStorage") : i1, !hal.buffer
    cf.assert %ok, "failed to import tokens"

    %num_tok = arith.divui %tok_bytes, %c8 : index
    %token_tensor = hal.tensor.import %ref : !hal.buffer -> tensor<?xi64>{%num_tok}

    return %token_tensor : tensor<?xi64>
  }
}
