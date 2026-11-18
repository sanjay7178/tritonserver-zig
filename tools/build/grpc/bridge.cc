#include "bridge.h"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>

#include "grpc/grpc_server.h"
#include "shared_memory_manager.h"

namespace {
using NativeFrontend = triton::server::grpc::Server;

[[noreturn]] void fatal_start(const char* message) noexcept
{
  // Upstream sets running_ only after launching all handlers. If Start throws
  // partway through that sequence, Stop cannot join the live handler threads.
  std::fprintf(stderr, "Triton gRPC startup failed: %s\n", message);
  std::fflush(stderr);
  std::_Exit(EXIT_FAILURE);
}
}  // namespace

struct TRITONZIG_GrpcFrontend {
  std::unique_ptr<NativeFrontend> service;
  bool shutdown_started = false;
};

TRITONSERVER_Error* tritonzig_grpc_start(
    TRITONSERVER_Server* server, const char* address, uint16_t port,
    TRITONZIG_GrpcFrontend** frontend) noexcept
{
  if (server == nullptr || address == nullptr || *address == '\0' || port == 0 ||
      frontend == nullptr || *frontend != nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "invalid Triton gRPC start arguments");
  }

  try {
    auto result = std::make_unique<TRITONZIG_GrpcFrontend>();
    // The Zig server owns this handle, as in Triton's in-process bindings.
    std::shared_ptr<TRITONSERVER_Server> borrowed(server, [](TRITONSERVER_Server*) {});
    auto shared_memory = std::make_shared<triton::server::SharedMemoryManager>(false);
    triton::server::grpc::Options options;
    options.socket_.address_ = address;
    options.socket_.port_ = port;
    if (auto* error = NativeFrontend::Create(
            borrowed, nullptr, shared_memory, options, &result->service)) {
      return error;
    }

    try {
      if (auto* error = result->service->Start()) {
        return error;
      }
    } catch (const std::exception& error) {
      fatal_start(error.what());
    } catch (...) {
      fatal_start("unknown native exception");
    }

    *frontend = result.release();
    return nullptr;
  } catch (const std::exception& error) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, error.what());
  } catch (...) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, "unknown exception creating Triton gRPC frontend");
  }
}

TRITONSERVER_Error* tritonzig_grpc_prepare_shutdown(
    TRITONZIG_GrpcFrontend* frontend, uint32_t* remaining_seconds) noexcept
{
  if (frontend == nullptr || remaining_seconds == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "invalid Triton gRPC shutdown arguments");
  }
  if (frontend->shutdown_started) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_ALREADY_EXISTS, "Triton gRPC shutdown has already started");
  }
  // GracefulStop creates a thread. A second call could replace a joinable thread.
  frontend->shutdown_started = true;

  try {
    return frontend->service->GracefulStop(remaining_seconds);
  } catch (const std::exception& error) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, error.what());
  } catch (...) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, "unknown exception preparing Triton gRPC shutdown");
  }
}

TRITONSERVER_Error* tritonzig_grpc_stop(TRITONZIG_GrpcFrontend** frontend) noexcept
{
  if (frontend == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "invalid Triton gRPC frontend slot");
  }
  if (*frontend == nullptr) {
    return nullptr;
  }

  try {
    if (auto* error = (*frontend)->service->Stop()) {
      return error;
    }
    delete *frontend;
    *frontend = nullptr;
    return nullptr;
  } catch (const std::exception& error) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, error.what());
  } catch (...) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, "unknown exception stopping Triton gRPC frontend");
  }
}
