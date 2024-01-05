// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include <sentencepiece_processor.h>

#include <cstdio>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#define IREE_STATUS_FEATURES 2

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/types.h"
#include "iree/vm/api.h"
#include "iree/vm/dynamic/api.h"
#include "iree/vm/native_module_cc.h"

// NOTE: this module is written in C++ using the native module wrapper and uses
// template magic to handle marshaling arguments. For a lot of uses this is a
// much friendlier way of exposing modules to the IREE VM and if performance and
// code size are not a concern is a fine route to take. Here we do it for
// brevity but all of the internal IREE modules are implemented in C.

//===----------------------------------------------------------------------===//
// !custom.string type
//===----------------------------------------------------------------------===//

// The "string" type we use to store and retain string data.
// This could be arbitrarily complex or simply wrap another user-defined type.
// The descriptor that is registered at startup defines how to manage the
// lifetime of the type (such as which destruction function is called, if any).
// See ref.h for more information and additional utilities.
typedef struct iree_tokenizer_spm_t {
  // Must be the first field; used to track the reference count of the object.
  iree_vm_ref_object_t ref_object;
  iree_allocator_t allocator;
  // SentencePiece class for storing the internal tokenizer state.
  sentencepiece::SentencePieceProcessor* tokenizer;
} iree_tokenizer_spm_t;

// Runtime type descriptor for the !custom.string describing how to manage it
// and destroy it. The type ID is allocated at runtime and does not need to
// match the compiler ID.
IREE_VM_DECLARE_TYPE_ADAPTERS(iree_tokenizer_spm, iree_tokenizer_spm_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_tokenizer_spm, iree_tokenizer_spm_t);

// Creates a new !tokenizer.spm object with a copy of the given |value|.
// Applications could use this and any other methods we wanted to expose to
// interop with the loaded VM modules - such as passing in/out the objects.
// We don't need this for the demo but creating the custom object, appending it
// to the invocation input list, and then consuming it in the compiled module
// is straightforward.
static iree_status_t iree_tokenizer_spm_create(
    std::string_view string, iree_allocator_t allocator,
    iree_tokenizer_spm_t** out_tokenizer) {
  IREE_ASSERT_ARGUMENT(out_tokenizer);
  iree_tokenizer_spm_t* wrapped_tokenizer = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      allocator, sizeof(*wrapped_tokenizer), (void**)&wrapped_tokenizer));
  wrapped_tokenizer->ref_object.counter = IREE_ATOMIC_VAR_INIT(1);
  wrapped_tokenizer->allocator = allocator;

  sentencepiece::SentencePieceProcessor* tokenizer =
      new sentencepiece::SentencePieceProcessor();
  const auto spmStatus = tokenizer->LoadFromSerializedProto(string);
  if (!spmStatus.ok()) {
    // return iree_make_status(IREE_STATUS_UNKNOWN,
    //                         "Failed to load tokenizer module ");
    return iree_make_status(IREE_STATUS_UNKNOWN);
  }
  if (!tokenizer->SetEncodeExtraOptions("bos:eos").ok()) {
    // return iree_make_status(IREE_STATUS_UNKNOWN,
    //                         "Failed to set extra tokenizer encode options");
    return iree_make_status(IREE_STATUS_UNKNOWN);
  }
  if (!tokenizer->SetDecodeExtraOptions("bos:eos").ok()) {
    // return iree_make_status(IREE_STATUS_UNKNOWN,
    //                         "Failed to set extra tokenizer decode options");
    return iree_make_status(IREE_STATUS_UNKNOWN);
  }
  wrapped_tokenizer->tokenizer = tokenizer;
  *out_tokenizer = wrapped_tokenizer;
  return iree_ok_status();
}

static void iree_tokenizer_spm_destroy(void* ptr) {
  iree_tokenizer_spm_t* wrapped_tokenizer = (iree_tokenizer_spm_t*)ptr;
  delete wrapped_tokenizer->tokenizer;
  iree_allocator_free(wrapped_tokenizer->allocator, ptr);
}

static iree_vm_ref_type_descriptor_t iree_tokenizer_spm_descriptor_storage = {
    0};

// Registers types provided by the tokenizer module.
// We must call this before any of our types can be resolved.
static iree_status_t iree_tokenizer_basic_register_types(
    iree_vm_instance_t* instance) {
  iree_tokenizer_spm_descriptor_storage.destroy = iree_tokenizer_spm_destroy;
  iree_tokenizer_spm_descriptor_storage.type_name = IREE_SV("tokenizer.spm");
  iree_tokenizer_spm_descriptor_storage.offsetof_counter =
      offsetof(iree_tokenizer_spm_t, ref_object.counter) /
      IREE_VM_REF_COUNTER_ALIGNMENT;
  return iree_vm_instance_register_type(instance,
                                        &iree_tokenizer_spm_descriptor_storage,
                                        &iree_tokenizer_spm_registration);
}

// Unregisters types previously registered.
// In dynamic modules it's critical that types are unregistered before the
// library is unloaded.
static void iree_tokenizer_basic_unregister_types(
    iree_vm_instance_t* instance) {
  iree_vm_instance_unregister_type(instance,
                                   &iree_tokenizer_spm_descriptor_storage);
}

//===----------------------------------------------------------------------===//
// VM module interface implementation
//===----------------------------------------------------------------------===//

namespace {

using namespace iree;

// Per-context module state.
// This can contain "globals" and other arbitrary state.
//
// Thread-compatible; the runtime will not issue multiple calls at the same
// time using the same state. If the implementation uses external threads then
// it must synchronize itself.
class TokenizerModuleState final {
 public:
  explicit TokenizerModuleState(iree_allocator_t host_allocator)
      : host_allocator_(host_allocator) {}
  ~TokenizerModuleState() = default;

  // Creates a new tokenizer based on the contents of the buffer.
  StatusOr<vm::ref<iree_tokenizer_spm_t>> LoadTokenizerFromTensor(
      vm::ref<iree_hal_buffer_view_t> buffer_view) {
    auto* view = buffer_view.get();
    iree_hal_buffer_t* buf = iree_hal_buffer_view_buffer(view);
    iree_device_size_t size = iree_hal_buffer_view_byte_length(view);
    iree_hal_buffer_mapping_t mapping = {{0}};
    IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
        buf, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
        /*byte_offset=*/0, /*byte_length=*/size, &mapping));

    vm::ref<iree_tokenizer_spm_t> tokenizer;
    IREE_RETURN_IF_ERROR(iree_tokenizer_spm_create(
        std::string_view(reinterpret_cast<const char*>(mapping.contents.data),
                         mapping.contents.data_length),
        host_allocator_, &tokenizer));
    IREE_RETURN_IF_ERROR(iree_hal_buffer_unmap_range(&mapping));
    return std::move(tokenizer);
  }

  // Creates a new tokenizer based on the contents of the buffer.
  StatusOr<vm::ref<iree_tokenizer_spm_t>> LoadTokenizerFromBuffer(
      vm::ref<iree_vm_buffer_t> buffer) {
    vm::ref<iree_tokenizer_spm_t> tokenizer;
    IREE_RETURN_IF_ERROR(iree_tokenizer_spm_create(
        std::string_view(reinterpret_cast<const char*>(buffer->data.data),
                         buffer->data.data_length),
        host_allocator_, &tokenizer));
    return std::move(tokenizer);
  }

  StatusOr<vm::ref<iree_vm_buffer_t>> EncodeI64(
      const vm::ref<iree_tokenizer_spm_t> tokenizer,
      vm::ref<iree_vm_buffer_t> line) {
    IREE_ASSERT_ARGUMENT(tokenizer);
    IREE_ASSERT_ARGUMENT(line);

    std::vector<int> ids;
    if (!tokenizer->tokenizer
             ->Encode(std::string_view(
                          reinterpret_cast<const char*>(line->data.data),
                          line->data.data_length),
                      &ids)
             .ok()) {
      // return iree_make_status(IREE_STATUS_UNKNOWN, "Failed to decode line");
      return iree_make_status(IREE_STATUS_UNKNOWN);
    }

    // SentencePiece can only populate i32 token widths so copy to the target
    // width.
    std::vector<int64_t> i64_ids(ids.begin(), ids.end());

    iree_vm_buffer_t* buffer = NULL;
    IREE_RETURN_IF_ERROR(iree_vm_buffer_create(
        IREE_VM_BUFFER_ACCESS_ORIGIN_GUEST | IREE_VM_BUFFER_ACCESS_MUTABLE,
        i64_ids.size() * sizeof(int64_t), 64, host_allocator_, &buffer));
    IREE_RETURN_IF_ERROR(iree_vm_buffer_write_elements(
        i64_ids.data(), buffer, 0, i64_ids.size(), sizeof(int64_t)));

    return vm::ref<iree_vm_buffer_t>(std::move(buffer));
  }

 private:
  // Allocator that the caller requested we use for any allocations we need to
  // perform during operation.
  iree_allocator_t host_allocator_;
};

// Function table mapping imported function names to their implementation.
static const vm::NativeFunction<TokenizerModuleState>
    kTokenizerModuleFunctions[] = {
        vm::MakeNativeFunction("load_spm.from_buffer",
                               &TokenizerModuleState::LoadTokenizerFromBuffer),
        vm::MakeNativeFunction("load_spm.from_tensor",
                               &TokenizerModuleState::LoadTokenizerFromTensor),
        vm::MakeNativeFunction("encode_i64", &TokenizerModuleState::EncodeI64),
};

// The module instance that will be allocated and reused across contexts.
// Any context-specific state must be stored in a TokenizerModuleState.
class TokenizerModule final : public vm::NativeModule<TokenizerModuleState> {
 public:
  using vm::NativeModule<TokenizerModuleState>::NativeModule;

  ~TokenizerModule() override {
    iree_tokenizer_basic_unregister_types(instance());
  }

  // Creates per-context state when the module is added to a new context.
  // May be called from any thread.
  StatusOr<std::unique_ptr<TokenizerModuleState>> CreateState(
      iree_allocator_t host_allocator) override {
    auto state = std::make_unique<TokenizerModuleState>(host_allocator);
    return state;
  }
};

}  // namespace

// Creates a native tokenizer module that can be reused in multiple contexts.
extern "C" IREE_VM_DYNAMIC_MODULE_EXPORT iree_status_t create_tokenizer_module(
    iree_vm_dynamic_module_version_t max_version, iree_vm_instance_t* instance,
    iree_host_size_t param_count, const iree_string_pair_t* params,
    iree_allocator_t host_allocator, iree_vm_module_t** out_module) {
  // Ensure the version matches; the version will change if the VM module
  // interface changes and existing libraries are incompatible.
  if (max_version != IREE_VM_DYNAMIC_MODULE_VERSION_LATEST) {
    // return iree_make_status(
    //     IREE_STATUS_UNIMPLEMENTED,
    //     "unsupported runtime version %u, module compiled with version %u",
    //     max_version, IREE_VM_DYNAMIC_MODULE_VERSION_LATEST);
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
  }

#if IREE_TRACING_FEATURES
  // Today Tracy cannot be used with custom dynamic modules as it'll try to
  // create a new tracing context distinct from the hosting application. Custom
  // module libraries should be built with tracing disabled.
  fprintf(stderr,
          "Tracy is not currently supported in custom dynamic modules\n");
#endif  // IREE_TRACING_FEATURES

  IREE_RETURN_IF_ERROR(iree_hal_module_resolve_all_types(instance));
  IREE_RETURN_IF_ERROR(iree_tokenizer_basic_register_types(instance));

  auto module = std::make_unique<TokenizerModule>(
      "tokenizer", /*version=*/0, instance, host_allocator,
      iree::span<const vm::NativeFunction<TokenizerModuleState>>(
          kTokenizerModuleFunctions));
  *out_module = module.release()->interface();
  return iree_ok_status();
}
