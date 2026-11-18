// Test-only CPU pass-through backend with execution/batch proof outputs.
// Deliberate errors and malformed output modes are confined to this fixture.
#include "triton/core/tritonbackend.h"

#include <atomic>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <thread>
#include <vector>

namespace {
std::atomic<int64_t> next_execution{1};

TRITONSERVER_Error* Invalid(const char* message)
{
  return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, message);
}

void Ignore(TRITONSERVER_Error* error)
{
  if (error != nullptr) TRITONSERVER_ErrorDelete(error);
}

struct Input {
  TRITONBACKEND_Input* handle = nullptr;
  std::vector<int64_t> shape;
  std::vector<uint8_t> bytes;
  uint64_t batch = 1;
};

TRITONSERVER_Error* ReadInput(
    TRITONBACKEND_Request* request, Input* result, const char* name = "IN0")
{
  auto error = TRITONBACKEND_RequestInput(request, name, &result->handle);
  if (error != nullptr) return error;
  TRITONSERVER_DataType datatype;
  const int64_t* shape = nullptr;
  uint32_t rank = 0;
  uint64_t byte_size = 0;
  uint32_t buffer_count = 0;
  error = TRITONBACKEND_InputProperties(
      result->handle, nullptr, &datatype, &shape, &rank, &byte_size, &buffer_count);
  if (error != nullptr) return error;
  if (datatype != TRITONSERVER_TYPE_INT32 || (rank != 1 && rank != 2))
    return Invalid("phase4 fixture requires rank-1/rank-2 INT32 input");
  result->shape.assign(shape, shape + rank);
  size_t elements = 1;
  for (const auto dim : result->shape) {
    if (dim < 0 || (elements != 0 &&
        uint64_t(dim) > std::numeric_limits<size_t>::max() / elements))
      return Invalid("phase4 fixture input shape overflow");
    elements *= size_t(dim);
  }
  if (elements > std::numeric_limits<size_t>::max() / sizeof(int32_t) ||
      elements * sizeof(int32_t) != byte_size)
    return Invalid("phase4 fixture input byte-size mismatch");
  result->batch = rank == 2 ? uint64_t(shape[0]) : 1;
  if (result->batch == 0) return Invalid("phase4 fixture requires a nonempty batch");
  result->bytes.resize(size_t(byte_size));
  size_t copied = 0;
  for (uint32_t i = 0; i < buffer_count; ++i) {
    const void* base = nullptr;
    uint64_t size = 0;
    TRITONSERVER_MemoryType memory = TRITONSERVER_MEMORY_CPU;
    int64_t memory_id = 0;
    error = TRITONBACKEND_InputBuffer(result->handle, i, &base, &size, &memory, &memory_id);
    if (error != nullptr) return error;
    if (memory != TRITONSERVER_MEMORY_CPU && memory != TRITONSERVER_MEMORY_CPU_PINNED)
      return Invalid("phase4 fixture supports CPU input only");
    if (size > byte_size - copied || (size != 0 && base == nullptr))
      return Invalid("phase4 fixture invalid input buffer");
    if (size != 0) std::memcpy(result->bytes.data() + copied, base, size_t(size));
    copied += size_t(size);
  }
  if (copied != byte_size) return Invalid("phase4 fixture incomplete input buffers");
  return nullptr;
}

TRITONSERVER_Error* Output(
    TRITONBACKEND_Response* response, const char* name,
    TRITONSERVER_DataType datatype, const std::vector<int64_t>& shape,
    const void* bytes, uint64_t byte_size)
{
  TRITONBACKEND_Output* output = nullptr;
  auto error = TRITONBACKEND_ResponseOutput(
      response, &output, name, datatype, shape.data(), uint32_t(shape.size()));
  if (error != nullptr) return error;
  void* buffer = nullptr;
  TRITONSERVER_MemoryType memory = TRITONSERVER_MEMORY_CPU;
  int64_t memory_id = 0;
  error = TRITONBACKEND_OutputBuffer(output, &buffer, byte_size, &memory, &memory_id);
  if (error != nullptr) return error;
  if (memory != TRITONSERVER_MEMORY_CPU || (byte_size != 0 && buffer == nullptr))
    return Invalid("phase4 fixture requires CPU output buffers");
  if (byte_size != 0) std::memcpy(buffer, bytes, size_t(byte_size));
  return nullptr;
}

TRITONSERVER_Error* Process(
    TRITONBACKEND_Request* request, TRITONBACKEND_Response* response,
    const Input& input, int64_t execution_id, int64_t total_batch,
    uint32_t request_count, bool malformed)
{
  if (input.bytes.size() >= sizeof(int32_t)) {
    int32_t first;
    std::memcpy(&first, input.bytes.data(), sizeof(first));
    if (first == INT32_MIN) return Invalid("intentional phase4 execution failure");
    // Keep one worker busy for bounded-queue and shutdown race tests.
    if (first == INT32_MAX) std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  uint32_t output_count = 0;
  auto error = TRITONBACKEND_RequestOutputCount(request, &output_count);
  if (error != nullptr) return error;
  for (uint32_t i = 0; i < output_count; ++i) {
    const char* name = nullptr;
    error = TRITONBACKEND_RequestOutputName(request, i, &name);
    if (error != nullptr) return error;
    if (std::strcmp(name, "OUT0") == 0) {
      if (!malformed) {
        error = Output(response, "OUT0", TRITONSERVER_TYPE_INT32,
                       input.shape, input.bytes.data(), input.bytes.size());
      } else {
        auto shape = input.shape;
        ++shape[0];
        const size_t extra = input.bytes.size() / size_t(input.batch);
        auto bytes = input.bytes;
        bytes.resize(bytes.size() + extra, 0);
        error = Output(response, "OUT0", TRITONSERVER_TYPE_INT32,
                       shape, bytes.data(), bytes.size());
      }
    } else if (std::strcmp(name, "OUT1") == 0) {
      Input second;
      error = ReadInput(request, &second, "IN1");
      if (error != nullptr) return error;
      if (second.batch != input.batch || second.shape != input.shape)
        return Invalid("phase4 fixture input shapes disagree");
      error = Output(response, "OUT1", TRITONSERVER_TYPE_INT32,
                     second.shape, second.bytes.data(), second.bytes.size());
    } else if (std::strcmp(name, "BATCH_INFO") == 0) {
      const auto batch = input.batch;
      std::vector<int64_t> values(size_t(batch) * 3);
      for (uint64_t row = 0; row < batch; ++row) {
        values[size_t(row) * 3] = execution_id;
        values[size_t(row) * 3 + 1] = total_batch;
        values[size_t(row) * 3 + 2] = request_count;
      }
      std::vector<int64_t> shape = input.shape.size() == 2
          ? std::vector<int64_t>{int64_t(batch), 3}
          : std::vector<int64_t>{3};
      error = Output(response, "BATCH_INFO", TRITONSERVER_TYPE_INT64,
                     shape, values.data(), values.size() * sizeof(int64_t));
    } else {
      return Invalid("phase4 fixture unknown requested output");
    }
    if (error != nullptr) return error;
  }
  return nullptr;
}
}  // namespace

extern "C" TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
try
{
  // Prepare every response before accepting ownership or sending any result.
  std::vector<TRITONBACKEND_Response*> responses(request_count, nullptr);
  std::vector<Input> inputs(request_count);
  std::vector<TRITONSERVER_Error*> failures(request_count, nullptr);
  for (uint32_t i = 0; i < request_count; ++i) {
    auto error = TRITONBACKEND_ResponseNew(&responses[i], requests[i]);
    if (error != nullptr) {
      for (auto* response : responses)
        if (response != nullptr) Ignore(TRITONBACKEND_ResponseDelete(response));
      return error;
    }
  }
  TRITONBACKEND_Model* model = nullptr;
  const char* model_name = "";
  Ignore(TRITONBACKEND_ModelInstanceModel(instance, &model));
  if (model != nullptr) Ignore(TRITONBACKEND_ModelName(model, &model_name));
  const bool malformed = std::strcmp(model_name, "phase4_malformed") == 0;
  const auto execution = next_execution.fetch_add(1, std::memory_order_relaxed);
  int64_t total_batch = 0;
  for (uint32_t i = 0; i < request_count; ++i) {
    try {
      failures[i] = ReadInput(requests[i], &inputs[i]);
      if (failures[i] == nullptr) total_batch += int64_t(inputs[i].batch);
    } catch (const std::bad_alloc&) {
      failures[i] = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, "phase4 fixture allocation failure");
    }
  }
  for (uint32_t i = 0; i < request_count; ++i) {
    auto* failure = failures[i];
    if (failure == nullptr) {
      try {
        failure = Process(requests[i], responses[i], inputs[i], execution,
                          total_batch, request_count, malformed);
      } catch (const std::bad_alloc&) {
        failure = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, "phase4 fixture allocation failure");
      }
    }
    Ignore(TRITONBACKEND_ResponseSend(
        responses[i], TRITONSERVER_RESPONSE_COMPLETE_FINAL, failure));
    Ignore(failure);
    Ignore(TRITONBACKEND_RequestRelease(requests[i], TRITONSERVER_REQUEST_RELEASE_ALL));
  }
  return nullptr;
}
catch (const std::bad_alloc&)
{
  // Only the initial vector allocations can reach here; no response has been
  // created and ownership of every request remains with Triton.
  return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, "phase4 fixture allocation failure");
}
