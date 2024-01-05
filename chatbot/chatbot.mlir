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
  // Tokenizer Imports
  //===--------------------------------------------------------------------===//
  // Creates a new tokenizer (on the host) with contents from the given tensor.
  // Probably don't use this.
  func.func private @tokenizer.load_spm.from_tensor(tensor<?xi8>) -> !tokenizer.spm

  // Creates a new tokenizer with the contents of the given buffer.
  func.func private @tokenizer.load_spm.from_buffer(!util.buffer) -> !tokenizer.spm

  // Returns an array of 64-bit tokens based on the tokenizer and given text.
  func.func private @tokenizer.encode_i64(!tokenizer.spm, !util.buffer) -> !util.buffer

  // Returns a string buffer of the text decoded from the given tokens.
  func.func private @tokenizer.decode_i64(!tokenizer.spm, tensor<1x?xi64>) -> !util.buffer

  // Returns a string buffer of the text decoded from the given tokens.
  func.func private @tokenizer.is_not_eos(!tokenizer.spm, tensor<1x1xi64>) -> i1

  //===--------------------------------------------------------------------===//
  // LLaMa Imports
  //===--------------------------------------------------------------------===//
  func.func private @state_update.run_initialize(%input: tensor<1x?xi64>) -> tensor<1x1xi64> attributes {
    iree.abi.model = "coarse-fences"
  }
  func.func private @state_update.run_forward(%input: tensor<1x1xi64>) -> tensor<1x1xi64> attributes {
    iree.abi.model = "coarse-fences"
  }

  func.func private @gen_text(%stdout: !io_stream.handle,
                              %tspm: !tokenizer.spm,
                              %input: tensor<1x?xi64>) -> tensor<1x?xi64> {
    %text = util.buffer.constant : !util.buffer = "Generating token..."
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c511 = arith.constant 511 : index

    %empty = tensor.empty() : tensor<1x512xi64>
    %first_token = func.call @state_update.run_initialize(%input)
                   : (tensor<1x?xi64>) -> tensor<1x1xi64>
    %init = tensor.insert_slice %first_token into
                     %empty[0, 0] [1, 1] [1, 1] : tensor<1x1xi64> into tensor<1x512xi64>

    %import_text = util.buffer.constant : !util.buffer = "Imported tokens..."
    func.call @print_buffer(%stdout, %import_text) : (!io_stream.handle, !util.buffer) -> ()

    // Loop until the end-of-sentence or maximum number of tokens is reached.
    %s, %toks, %last_tok = scf.while(%step = %c0,
                                     %tokens = %init,
                                     %prev_tok = %first_token)
    : (index, tensor<1x512xi64>, tensor<1x1xi64>) -> (index, tensor<1x512xi64>, tensor<1x1xi64>) {
      %is_not_eos = func.call @tokenizer.is_not_eos(%tspm, %prev_tok)
                    : (!tokenizer.spm, tensor<1x1xi64>) -> i1
      %cond_in_range = arith.cmpi slt, %step, %c511 : index
      %continue = arith.andi %cond_in_range, %is_not_eos : i1
      %stepp1 = arith.addi %step, %c1 : index
      scf.condition(%continue) %stepp1, %tokens, %prev_tok
                               : index, tensor<1x512xi64>, tensor<1x1xi64>
    } do {
    ^bb0(%curr: index, %curr_tokens: tensor<1x512xi64>, %prev: tensor<1x1xi64>):
      func.call @print_buffer(%stdout, %text) : (!io_stream.handle, !util.buffer) -> ()
      %next = func.call @state_update.run_forward(%prev) : (tensor<1x1xi64>) -> tensor<1x1xi64>
      %next_text = tensor.insert_slice %next into
                     %curr_tokens[0, %curr] [1, 1] [1, 1] : tensor<1x1xi64> into tensor<1x512xi64>
      scf.yield %curr, %next_text, %next : index, tensor<1x512xi64>, tensor<1x1xi64>
    }

    %final_text = tensor.extract_slice %toks[0, 0] [1, %s] [1, 1] : tensor<1x512xi64> to tensor<1x?xi64>
    return %final_text : tensor<1x?xi64>
  }

  //===--------------------------------------------------------------------===//
  // Entry Point
  //===--------------------------------------------------------------------===//
  func.func @chat() -> () {
    %stdin = io_stream.console.stdin : !io_stream.handle
    %stdout = io_stream.console.stdout : !io_stream.handle
  
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c512 = arith.constant 512 : index
    %true = arith.constant 1 : i1
    %null = util.null : !util.buffer

    // We need to get the device allocator to import the util buffer.
    %device_0 = hal.devices.get %c0 : !hal.device
    %allocator = hal.device.allocator<%device_0 : !hal.device> : !hal.allocator

    %affinity = arith.constant -1 : i64

    %serialized_tokenizer = util.global.load @serialized_tokenizer : tensor<499723xi8>
    %cast = tensor.cast %serialized_tokenizer : tensor<499723xi8> to tensor<?xi8>
    %tokenizer = func.call @tokenizer.load_spm.from_tensor(%cast) : (tensor<?xi8>) -> !tokenizer.spm
  
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
          %tokens = func.call @tokenizer.encode_i64(%tokenizer, %line) : (!tokenizer.spm, !util.buffer) -> !util.buffer
          %tok_bytes = util.buffer.size %tokens : !util.buffer
          %ok, %ref = hal.allocator.import<%allocator : !hal.allocator>
                  source(%tokens : !util.buffer)[%c0, %tok_bytes]
                  affinity(%affinity) type(DeviceLocal)
                  usage("TransferSource|TransferTarget|Transfer|DispatchStorage") : i1, !hal.buffer
          cf.assert %ok, "failed to import tokens"

          %num_tok = arith.divui %tok_bytes, %c8 : index
          %token_tensor = hal.tensor.import %ref : !hal.buffer -> tensor<1x?xi64>{%num_tok}

          %import_text = util.buffer.constant : !util.buffer = "Imported tokens..."
          func.call @print_buffer(%stdout, %import_text) : (!io_stream.handle, !util.buffer) -> ()

          %new_line = func.call @gen_text(%stdout, %tokenizer, %token_tensor)
                      : (!io_stream.handle, !tokenizer.spm, tensor<1x?xi64>) -> tensor<1x?xi64>
          %detok_line = func.call @tokenizer.decode_i64(%tokenizer, %new_line)
                        : (!tokenizer.spm, tensor<1x?xi64>) -> !util.buffer
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
