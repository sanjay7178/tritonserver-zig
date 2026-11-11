// Optional synthetic GPU workload. Triton transports CPU tensors, while this
// backend explicitly runs a dense FP32 network on CUDA device zero using cuBLAS.
// Uniform normalized weights make every output equal the input sample's mean.
#include "triton/core/tritonbackend.h"

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <string>
#include <vector>

namespace {
constexpr int kWidth = 64;
constexpr int kMaxBatch = 64;

TRITONSERVER_Error* Error(const std::string& message)
{
  return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, message.c_str());
}

TRITONSERVER_Error* Invalid(const char* message)
{
  return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, message);
}

TRITONSERVER_Error* Cuda(cudaError_t status, const char* operation)
{
  return status == cudaSuccess ? nullptr
      : Error(std::string(operation) + ": " + cudaGetErrorString(status));
}

TRITONSERVER_Error* Blas(cublasStatus_t status, const char* operation)
{
  return status == CUBLAS_STATUS_SUCCESS ? nullptr
      : Error(std::string(operation) + ": " + cublasGetStatusString(status));
}

#define RETURN_IF_ERROR(expression) \
  do { \
    if (TRITONSERVER_Error* error = (expression)) return error; \
  } while (false)

void LogAndDelete(TRITONSERVER_Error* error)
{
  if (error != nullptr) {
    std::fprintf(stderr, "gpu_batch: %s\n", TRITONSERVER_ErrorMessage(error));
    TRITONSERVER_ErrorDelete(error);
  }
}

bool EnvironmentInteger(const char* name, int fallback, int minimum, int maximum, int* value)
{
  const char* text = std::getenv(name);
  if (text == nullptr) {
    *value = fallback;
    return true;
  }
  char* end = nullptr;
  errno = 0;
  const long parsed = std::strtol(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || parsed < minimum || parsed > maximum)
    return false;
  *value = static_cast<int>(parsed);
  return true;
}

struct DeviceState {
  int hidden = 4096;
  int layers = 16;
  cudaStream_t stream = nullptr;
  cublasHandle_t blas = nullptr;
  float* projection = nullptr;
  float* weights = nullptr;
  float* reduction = nullptr;
  float* input = nullptr;
  float* hidden_a = nullptr;
  float* hidden_b = nullptr;
  float* output = nullptr;
  std::vector<float> host_input;

  ~DeviceState()
  {
    if (stream != nullptr || blas != nullptr) {
      (void)cudaSetDevice(0);
      if (stream != nullptr) (void)cudaStreamSynchronize(stream);
    }
    if (blas != nullptr) (void)cublasDestroy(blas);
    for (float* allocation : {projection, weights, reduction, input, hidden_a, hidden_b, output})
      if (allocation != nullptr) (void)cudaFree(allocation);
    if (stream != nullptr) (void)cudaStreamDestroy(stream);
  }

  TRITONSERVER_Error* Allocate(float** destination, size_t elements)
  {
    return Cuda(cudaMalloc(reinterpret_cast<void**>(destination), elements * sizeof(float)), "cudaMalloc");
  }

  TRITONSERVER_Error* Uniform(float** destination, size_t elements, float value)
  {
    RETURN_IF_ERROR(Allocate(destination, elements));
    const std::vector<float> host(elements, value);
    return Cuda(cudaMemcpy(*destination, host.data(), elements * sizeof(float), cudaMemcpyHostToDevice), "upload weights");
  }

  TRITONSERVER_Error* Initialize()
  {
    if (!EnvironmentInteger("GPU_EXPERIMENT_HIDDEN", 4096, 64, 8192, &hidden) ||
        (hidden & (hidden - 1)) != 0)
      return Invalid("GPU_EXPERIMENT_HIDDEN must be a power of two between 64 and 8192");
    if (!EnvironmentInteger("GPU_EXPERIMENT_LAYERS", 16, 1, 256, &layers))
      return Invalid("GPU_EXPERIMENT_LAYERS must be between 1 and 256");
    RETURN_IF_ERROR(Cuda(cudaSetDevice(0), "cudaSetDevice(0)"));
    cudaDeviceProp device{};
    RETURN_IF_ERROR(Cuda(cudaGetDeviceProperties(&device, 0), "cudaGetDeviceProperties"));
    RETURN_IF_ERROR(Cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate"));
    RETURN_IF_ERROR(Blas(cublasCreate(&blas), "cublasCreate"));
    RETURN_IF_ERROR(Blas(cublasSetStream(blas, stream), "cublasSetStream"));
    RETURN_IF_ERROR(Blas(cublasSetMathMode(blas, CUBLAS_PEDANTIC_MATH), "cublasSetMathMode"));
    RETURN_IF_ERROR(Uniform(&projection, static_cast<size_t>(hidden) * kWidth, 1.0f / kWidth));
    RETURN_IF_ERROR(Uniform(&weights, static_cast<size_t>(hidden) * hidden, 1.0f / hidden));
    RETURN_IF_ERROR(Uniform(&reduction, static_cast<size_t>(kWidth) * hidden, 1.0f / hidden));
    RETURN_IF_ERROR(Allocate(&input, kMaxBatch * kWidth));
    RETURN_IF_ERROR(Allocate(&hidden_a, static_cast<size_t>(kMaxBatch) * hidden));
    RETURN_IF_ERROR(Allocate(&hidden_b, static_cast<size_t>(kMaxBatch) * hidden));
    RETURN_IF_ERROR(Allocate(&output, kMaxBatch * kWidth));
    host_input.resize(kMaxBatch * kWidth);
    std::fprintf(stderr,
        "gpu_batch: CUDA device 0=%s; hidden=%d layers=%d max_batch=%d; FP32 pedantic cuBLAS; CPU Triton transport\n",
        device.name, hidden, layers, kMaxBatch);
    return nullptr;
  }

  TRITONSERVER_Error* Run(TRITONBACKEND_Request* request, TRITONBACKEND_Response* response)
  {
    uint32_t input_count = 0;
    RETURN_IF_ERROR(TRITONBACKEND_RequestInputCount(request, &input_count));
    if (input_count != 1) return Invalid("gpu_batch expects exactly one input named IN0");
    TRITONBACKEND_Input* tensor = nullptr;
    RETURN_IF_ERROR(TRITONBACKEND_RequestInput(request, "IN0", &tensor));
    TRITONSERVER_DataType datatype;
    const int64_t* shape = nullptr;
    uint32_t dimensions = 0, buffers = 0;
    uint64_t bytes = 0;
    RETURN_IF_ERROR(TRITONBACKEND_InputProperties(tensor, nullptr, &datatype, &shape, &dimensions, &bytes, &buffers));
    if (datatype != TRITONSERVER_TYPE_FP32 || dimensions != 2 || shape == nullptr ||
        shape[0] < 1 || shape[0] > kMaxBatch || shape[1] != kWidth ||
        bytes != static_cast<uint64_t>(shape[0] * kWidth * sizeof(float)))
      return Invalid("IN0 must be FP32 [batch,64] with batch between 1 and 64");
    const int batch = static_cast<int>(shape[0]);
    size_t copied = 0;
    for (uint32_t index = 0; index < buffers; ++index) {
      const void* buffer = nullptr;
      uint64_t length = 0;
      TRITONSERVER_MemoryType memory = TRITONSERVER_MEMORY_CPU;
      int64_t memory_id = 0;
      RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(tensor, index, &buffer, &length, &memory, &memory_id));
      if ((memory != TRITONSERVER_MEMORY_CPU && memory != TRITONSERVER_MEMORY_CPU_PINNED) ||
          length > bytes - copied || (length != 0 && buffer == nullptr))
        return Invalid("IN0 requires valid CPU input buffers");
      if (length != 0)
        std::memcpy(reinterpret_cast<char*>(host_input.data()) + copied, buffer, static_cast<size_t>(length));
      copied += static_cast<size_t>(length);
    }
    if (copied != bytes) return Invalid("IN0 input buffers do not cover its full shape");

    uint32_t output_count = 0;
    RETURN_IF_ERROR(TRITONBACKEND_RequestOutputCount(request, &output_count));
    if (output_count > 1) return Invalid("gpu_batch only provides OUT0");
    if (output_count == 1) {
      const char* name = nullptr;
      RETURN_IF_ERROR(TRITONBACKEND_RequestOutputName(request, 0, &name));
      if (name == nullptr || std::strcmp(name, "OUT0") != 0) return Invalid("gpu_batch only provides OUT0");
    }
    TRITONBACKEND_Output* tensor_output = nullptr;
    const int64_t output_shape[] = {batch, kWidth};
    RETURN_IF_ERROR(TRITONBACKEND_ResponseOutput(response, &tensor_output, "OUT0", TRITONSERVER_TYPE_FP32, output_shape, 2));
    void* host_output = nullptr;
    TRITONSERVER_MemoryType memory = TRITONSERVER_MEMORY_CPU;
    int64_t memory_id = 0;
    RETURN_IF_ERROR(TRITONBACKEND_OutputBuffer(tensor_output, &host_output, bytes, &memory, &memory_id));
    if (host_output == nullptr || (memory != TRITONSERVER_MEMORY_CPU && memory != TRITONSERVER_MEMORY_CPU_PINNED))
      return Invalid("gpu_batch requires a CPU output allocation");

    // Every queued CUDA command completes before response/request ownership is
    // returned, including a partial execution that fails below.
    struct Drain {
      cudaStream_t stream;
      ~Drain() { (void)cudaStreamSynchronize(stream); }
    } drain{stream};
    RETURN_IF_ERROR(Cuda(cudaMemcpyAsync(input, host_input.data(), bytes, cudaMemcpyHostToDevice, stream), "copy input to GPU"));
    const float alpha = 1.0f, beta = 0.0f;
    // Column-major [64,batch] is the same physical layout as [batch,64].
    RETURN_IF_ERROR(Blas(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N,
        hidden, batch, kWidth, &alpha, projection, hidden, input, kWidth,
        &beta, hidden_a, hidden), "input projection SGEMM"));
    float* current = hidden_a;
    float* next = hidden_b;
    for (int layer = 0; layer < layers; ++layer) {
      RETURN_IF_ERROR(Blas(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N,
          hidden, batch, hidden, &alpha, weights, hidden, current, hidden,
          &beta, next, hidden), "hidden SGEMM"));
      std::swap(current, next);
    }
    RETURN_IF_ERROR(Blas(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N,
        kWidth, batch, hidden, &alpha, reduction, kWidth, current, hidden,
        &beta, output, kWidth), "output reduction SGEMM"));
    RETURN_IF_ERROR(Cuda(cudaMemcpyAsync(host_output, output, bytes, cudaMemcpyDeviceToHost, stream), "copy output from GPU"));
    return Cuda(cudaStreamSynchronize(stream), "finish GPU inference");
  }
};
}  // namespace

#define GPU_BACKEND_EXPORT extern "C" __attribute__((visibility("default")))

GPU_BACKEND_EXPORT TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  try {
    auto state = std::make_unique<DeviceState>();
    RETURN_IF_ERROR(state->Initialize());
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(instance, state.get()));
    (void)state.release();
    return nullptr;
  } catch (const std::exception& error) {
    return Error(error.what());
  }
}

GPU_BACKEND_EXPORT TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* raw = nullptr;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &raw));
  delete static_cast<DeviceState*>(raw);
  return TRITONBACKEND_ModelInstanceSetState(instance, nullptr);
}

GPU_BACKEND_EXPORT TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(TRITONBACKEND_ModelInstance* instance,
    TRITONBACKEND_Request** requests, const uint32_t request_count)
{
  if (request_count != 1)
    return Invalid("gpu_batch expects one explicit batch; native request aggregation must be disabled");
  void* raw = nullptr;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &raw));
  if (raw == nullptr) return Error("GPU instance state is unavailable");
  RETURN_IF_ERROR(Cuda(cudaSetDevice(0), "select GPU for execution thread"));
  std::vector<TRITONBACKEND_Response*> responses;
  try {
    responses.resize(request_count, nullptr);
  } catch (const std::bad_alloc&) {
    return Error("response bookkeeping allocation failed");
  }
  // Create every response before taking ownership of any request. If creation
  // fails, returning an execute error leaves all requests with Triton.
  for (uint32_t index = 0; index < request_count; ++index) {
    if (TRITONSERVER_Error* error = TRITONBACKEND_ResponseNew(&responses[index], requests[index])) {
      for (TRITONBACKEND_Response* response : responses)
        if (response != nullptr) LogAndDelete(TRITONBACKEND_ResponseDelete(response));
      return error;
    }
  }
  for (uint32_t index = 0; index < request_count; ++index) {
    TRITONSERVER_Error* error = nullptr;
    try {
      error = static_cast<DeviceState*>(raw)->Run(requests[index], responses[index]);
    } catch (const std::exception& exception) {
      error = Error(exception.what());
    }
    LogAndDelete(TRITONBACKEND_ResponseSend(responses[index], TRITONSERVER_RESPONSE_COMPLETE_FINAL, error));
    if (error != nullptr) TRITONSERVER_ErrorDelete(error);
    LogAndDelete(TRITONBACKEND_RequestRelease(requests[index], TRITONSERVER_REQUEST_RELEASE_ALL));
  }
  return nullptr;
}
