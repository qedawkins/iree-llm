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
  func.func private @tokenizer.load_spm.from_tensor(!hal.device, tensor<?xi8>) -> !tokenizer.spm

  // Creates a new tokenizer with the contents of the given buffer.
  func.func private @tokenizer.load_spm.from_buffer(!util.buffer) -> !tokenizer.spm

  // Returns an array of 64-bit tokens based on the tokenizer and given text.
  func.func private @tokenizer.encode_i64(!tokenizer.spm, !hal.device, !util.buffer) -> (i32, !hal.buffer)

  // Returns a string buffer of the text decoded from the given tokens.
  func.func private @tokenizer.decode_i64(!tokenizer.spm, !hal.device, tensor<?xi64>) -> !util.buffer

  //===--------------------------------------------------------------------===//
  // Sample methods
  //===--------------------------------------------------------------------===//

  // CHECK-LABEL: EXEC @main
  func.func @main() -> tensor<?xi64> {
    %stdout = io_stream.console.stdout : !io_stream.handle
    %text = util.buffer.constant : !util.buffer = "please tokenize me :D"
    func.call @print_buffer(%stdout, %text) : (!io_stream.handle, !util.buffer) -> ()

    // We need to get the device allocator to import the util buffer.
    %c0 = arith.constant 0 : index
    %device_0 = hal.devices.get %c0 : !hal.device
    %allocator = hal.device.allocator<%device_0 : !hal.device> : !hal.allocator

    %c8 = arith.constant 8 : index
    %serialized_tokenizer = util.global.load @serialized_tokenizer : tensor<499723xi8>
    %cast = tensor.cast %serialized_tokenizer : tensor<499723xi8> to tensor<?xi8>
    %tokenizer = call @tokenizer.load_spm.from_tensor(%device_0, %cast) : (!hal.device, tensor<?xi8>) -> !tokenizer.spm

    %num_tok_i32, %tokens = call @tokenizer.encode_i64(%tokenizer, %device_0, %text) : (!tokenizer.spm, !hal.device, !util.buffer) -> (i32, !hal.buffer)

    %num_tok = arith.index_cast %num_tok_i32 : i32 to index
    %token_tensor = hal.tensor.import %tokens : !hal.buffer -> tensor<?xi64>{%num_tok}

    %detok_line = call @tokenizer.decode_i64(%tokenizer, %device_0, %token_tensor) : (!tokenizer.spm, !hal.device, tensor<?xi64>) -> !util.buffer
    func.call @print_buffer(%stdout, %detok_line) : (!io_stream.handle, !util.buffer) -> ()

    return %token_tensor : tensor<?xi64>
  }
}
