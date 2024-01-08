// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include <sentencepiece_processor.h>

#include <cstdio>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

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
    return iree_make_status(IREE_STATUS_UNKNOWN,
                            "Failed to load tokenizer module ");
  }
  if (!tokenizer->SetEncodeExtraOptions("bos").ok()) {
    return iree_make_status(IREE_STATUS_UNKNOWN,
                            "Failed to set extra tokenizer encode options");
  }
  if (!tokenizer->SetDecodeExtraOptions("bos").ok()) {
    return iree_make_status(IREE_STATUS_UNKNOWN,
                            "Failed to set extra tokenizer decode options");
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
      vm::ref<iree_hal_device_t> device,
      vm::ref<iree_hal_buffer_view_t> buffer_view) {
    auto* view = buffer_view.get();
    iree_hal_buffer_t* buf = iree_hal_buffer_view_buffer(view);
    iree_device_size_t size = iree_hal_buffer_view_byte_length(view);

    std::vector<uint8_t> actual_data(size);
    IREE_RETURN_IF_ERROR(iree_hal_device_transfer_d2h(
        device.get(), buf, /*source_offset=*/0,
        /*target_buffer=*/actual_data.data(),
        /*data_length=*/size, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
        iree_infinite_timeout()));

    vm::ref<iree_tokenizer_spm_t> tokenizer;
    IREE_RETURN_IF_ERROR(iree_tokenizer_spm_create(
        std::string_view(reinterpret_cast<const char*>(actual_data.data()),
                         size),
        host_allocator_, &tokenizer));
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

  StatusOr<std::tuple<int32_t, vm::ref<iree_hal_buffer_t>>> EncodeI64(
      const vm::ref<iree_tokenizer_spm_t> tokenizer,
      vm::ref<iree_hal_device_t> device, vm::ref<iree_vm_buffer_t> line) {
    IREE_ASSERT_ARGUMENT(tokenizer);
    IREE_ASSERT_ARGUMENT(line);

    std::vector<int> ids;
    if (!tokenizer->tokenizer
             ->Encode(std::string_view(
                          reinterpret_cast<const char*>(line->data.data),
                          line->data.data_length),
                      &ids)
             .ok()) {
      return iree_make_status(IREE_STATUS_UNKNOWN, "Failed to decode line");
    }

    // SentencePiece can only populate i32 token widths so copy to the target
    // width.
    std::vector<int64_t> i64_ids(ids.begin(), ids.end());

    iree_hal_buffer_t* hal_buffer = nullptr;
    const iree_hal_buffer_params_t params = {
        .usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                 IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE,
        .type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
    };
    IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
        iree_hal_device_allocator(device.get()), params,
        i64_ids.size() * sizeof(int64_t), &hal_buffer));
    IREE_RETURN_IF_ERROR(iree_hal_device_transfer_h2d(
        device.get(), i64_ids.data(), hal_buffer, 0,
        i64_ids.size() * sizeof(int64_t), IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
        iree_infinite_timeout()));

    return std::make_tuple(static_cast<int32_t>(i64_ids.size()),
                           vm::ref<iree_hal_buffer_t>(std::move(hal_buffer)));
  }

  StatusOr<vm::ref<iree_vm_buffer_t>> DecodeI64(
      const vm::ref<iree_tokenizer_spm_t> tokenizer,
      vm::ref<iree_hal_device_t> device,
      vm::ref<iree_hal_buffer_view_t> buffer_view) {
    IREE_ASSERT_ARGUMENT(tokenizer);
    IREE_ASSERT_ARGUMENT(buffer_view);

    auto* view = buffer_view.get();
    iree_hal_element_type_t element_type =
        iree_hal_buffer_view_element_type(view);
    if (!iree_hal_element_numerical_type_is_opaque(element_type) &&
        !iree_hal_element_numerical_type_is_integer(element_type) &&
        iree_hal_element_bit_count(element_type) != 64) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Invalid token element type");
    }

    iree_hal_buffer_t* buf = iree_hal_buffer_view_buffer(view);
    iree_device_size_t size = iree_hal_buffer_view_byte_length(view);
    std::vector<uint8_t> actual_data(size);
    IREE_RETURN_IF_ERROR(iree_hal_device_transfer_d2h(
        device.get(), buf, /*source_offset=*/0,
        /*target_buffer=*/actual_data.data(),
        /*data_length=*/size, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
        iree_infinite_timeout()));

    const int64_t* data = reinterpret_cast<const int64_t*>(actual_data.data());
    size_t num_tokens = static_cast<size_t>(size) / sizeof(int64_t);
    // std::vector<int> ids(num_tokens);
    // ids.assign(data, data + num_tokens);
    std::vector<int> ids;
    for (int i = 0, e = num_tokens; i < e; ++i) {
      ids.push_back(data[i]);
    }

    std::string detok;
    if (!tokenizer->tokenizer->Decode(ids, &detok).ok()) {
      return iree_make_status(IREE_STATUS_UNKNOWN, "Failed to decode line");
    }

    iree_vm_buffer_t* buffer = NULL;
    IREE_RETURN_IF_ERROR(iree_vm_buffer_create(
        IREE_VM_BUFFER_ACCESS_ORIGIN_GUEST | IREE_VM_BUFFER_ACCESS_MUTABLE,
        detok.size(), 1, host_allocator_, &buffer));
    IREE_RETURN_IF_ERROR(iree_vm_buffer_write_elements(detok.data(), buffer, 0,
                                                       detok.size(), 1));

    return vm::ref<iree_vm_buffer_t>(std::move(buffer));
  }

  StatusOr<int32_t> IsNotEOS(const vm::ref<iree_tokenizer_spm_t> tokenizer,
                             vm::ref<iree_hal_device_t> device,
                             vm::ref<iree_hal_buffer_view_t> buffer_view) {
    IREE_ASSERT_ARGUMENT(buffer_view);
    IREE_ASSERT_ARGUMENT(tokenizer);

    auto* view = buffer_view.get();
    iree_hal_element_type_t element_type =
        iree_hal_buffer_view_element_type(view);
    if (!iree_hal_element_numerical_type_is_opaque(element_type) &&
        !iree_hal_element_numerical_type_is_integer(element_type) &&
        iree_hal_element_bit_count(element_type) != 64) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Invalid token element type");
    }

    iree_device_size_t size = iree_hal_buffer_view_byte_length(view);
    if (size != 8) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Requires single token");
    }
    iree_hal_buffer_t* buf = iree_hal_buffer_view_buffer(view);
    iree_hal_buffer_mapping_t mapping = {{0}};
    std::vector<uint8_t> actual_data(size);
    IREE_RETURN_IF_ERROR(iree_hal_device_transfer_d2h(
        device.get(), buf, /*source_offset=*/0,
        /*target_buffer=*/actual_data.data(),
        /*data_length=*/size, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
        iree_infinite_timeout()));

    int64_t data = reinterpret_cast<const int64_t*>(actual_data.data())[0];
    return data != tokenizer->tokenizer->eos_id();
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
        vm::MakeNativeFunction("decode_i64", &TokenizerModuleState::DecodeI64),
        vm::MakeNativeFunction("is_not_eos", &TokenizerModuleState::IsNotEOS),
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
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "unsupported runtime version %u, module compiled with version %u",
        max_version, IREE_VM_DYNAMIC_MODULE_VERSION_LATEST);
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
