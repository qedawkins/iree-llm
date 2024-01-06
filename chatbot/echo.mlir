module @chatbot {

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

  func.func @chat() -> () {
    %stdin = io_stream.console.stdin : !io_stream.handle
    %stdout = io_stream.console.stdout : !io_stream.handle
  
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %true = arith.constant 1 : i1
    %null = util.null : !util.buffer

    // We need to get the device allocator to import the util buffer.
    %device_0 = hal.devices.get %c0 : !hal.device
    %allocator = hal.device.allocator<%device_0 : !hal.device> : !hal.allocator

    %affinity = arith.constant -1 : i64

    %serialized_tokenizer = util.global.load @serialized_tokenizer : tensor<499723xi8>
    %cast = tensor.cast %serialized_tokenizer : tensor<499723xi8> to tensor<?xi8>
    %tokenizer = func.call @tokenizer.load_spm.from_tensor(%device_0, %cast) : (!hal.device, tensor<?xi8>) -> !tokenizer.spm
  
    // Prompt printed each time we wait for input.
    %prompt = util.buffer.constant : !util.buffer = "type a line, ctrl-c to exit > "
  
    // Loop until the end-of-stream is reached.
    scf.while(%not_eos = %true) : (i1) -> i1 {
      scf.condition(%not_eos) %not_eos : i1
    } do {
    ^bb0(%_: i1):
      // Write prompt and read input until newline/ctrl-c is reached.
      io_stream.write.bytes(%stdout, %prompt) : (!io_stream.handle, !util.buffer) -> ()
      %line = io_stream.read.line(%stdin) : (!io_stream.handle) -> !util.buffer
      // A null return indicates end-of-stream.
      %not_eos = util.cmp.ne %line, %null : !util.buffer
      scf.if %not_eos {
        // If not yet at the end-of-stream check the returned line. If it's empty
        // (user just pressed enter) we skip it. stdin piped from files/echo/etc
        // will usually have a trailing newline.
        %line_length = util.buffer.size %line : !util.buffer
        %not_line_empty = arith.cmpi ne, %line_length, %c0 : index
        scf.if %not_line_empty {
          // Tokenize the line.
          %num_tok_i32, %tokens = func.call @tokenizer.encode_i64(%tokenizer, %device_0, %line) : (!tokenizer.spm, !hal.device, !util.buffer) -> (i32, !hal.buffer)
          %num_tok = arith.index_cast %num_tok_i32 : i32 to index
          %token_tensor = hal.tensor.import %tokens : !hal.buffer -> tensor<?xi64>{%num_tok}
          %detok_line = func.call @tokenizer.decode_i64(%tokenizer, %device_0, %token_tensor) : (!tokenizer.spm, !hal.device, tensor<?xi64>) -> !util.buffer
          func.call @print_buffer(%stdout, %detok_line) : (!io_stream.handle, !util.buffer) -> ()
          scf.yield
        }
        scf.yield
      }
      // Continue so long as we are not at the stdin end-of-stream.
      scf.yield %not_eos : i1
    }
  
    return
  }

}
