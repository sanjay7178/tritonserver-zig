#ifndef TRITONZIG_GRPC_BRIDGE_H
#define TRITONZIG_GRPC_BRIDGE_H

#include <stdint.h>
#include "triton/core/tritonserver.h"

#ifdef __cplusplus
extern "C" {
#define TRITONZIG_GRPC_NOEXCEPT noexcept
#else
#define TRITONZIG_GRPC_NOEXCEPT
#endif

typedef struct TRITONZIG_GrpcFrontend TRITONZIG_GrpcFrontend;

/* Borrows server until stop succeeds. The caller owns all returned errors.
 * Pass a null-initialized frontend slot and a host address, with IPv6 brackets
 * when applicable. */
struct TRITONSERVER_Error* tritonzig_grpc_start(
    struct TRITONSERVER_Server* server, const char* address, uint16_t port,
    TRITONZIG_GrpcFrontend** frontend) TRITONZIG_GRPC_NOEXCEPT;

/* Stops admission and spends part of the caller's in/out timeout on RPCs. */
struct TRITONSERVER_Error* tritonzig_grpc_prepare_shutdown(
    TRITONZIG_GrpcFrontend* frontend, uint32_t* remaining_seconds)
    TRITONZIG_GRPC_NOEXCEPT;

/* After the core stops, join the handlers, destroy the frontend and clear the
 * slot. A failed stop retains the frontend. */
struct TRITONSERVER_Error* tritonzig_grpc_stop(
    TRITONZIG_GrpcFrontend** frontend) TRITONZIG_GRPC_NOEXCEPT;

#undef TRITONZIG_GRPC_NOEXCEPT

#ifdef __cplusplus
}
#endif

#endif
