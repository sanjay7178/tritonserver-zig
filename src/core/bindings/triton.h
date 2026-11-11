// tritonbackend.h uses C++'s implicit struct-name typedefs even though its
// functions have C linkage. Supply only those opaque aliases for translate-c;
// every enum, function signature and ABI constant still comes from upstream.
#ifndef TRITONSERVER_ZIG_HEADERS_H
#define TRITONSERVER_ZIG_HEADERS_H
typedef struct TRITONSERVER_BufferAttributes TRITONSERVER_BufferAttributes;
typedef struct TRITONSERVER_Error TRITONSERVER_Error;
typedef struct TRITONSERVER_InferenceRequest TRITONSERVER_InferenceRequest;
typedef struct TRITONSERVER_InferenceResponse TRITONSERVER_InferenceResponse;
typedef struct TRITONSERVER_InferenceTrace TRITONSERVER_InferenceTrace;
typedef struct TRITONSERVER_Message TRITONSERVER_Message;
typedef struct TRITONSERVER_Metrics TRITONSERVER_Metrics;
typedef struct TRITONSERVER_Parameter TRITONSERVER_Parameter;
typedef struct TRITONSERVER_ResponseAllocator TRITONSERVER_ResponseAllocator;
typedef struct TRITONSERVER_Server TRITONSERVER_Server;
typedef struct TRITONSERVER_ServerOptions TRITONSERVER_ServerOptions;
typedef struct TRITONSERVER_Metric TRITONSERVER_Metric;
typedef struct TRITONSERVER_MetricFamily TRITONSERVER_MetricFamily;
typedef struct TRITONSERVER_MetricArgs TRITONSERVER_MetricArgs;
typedef struct TRITONBACKEND_MemoryManager TRITONBACKEND_MemoryManager;
typedef struct TRITONBACKEND_Input TRITONBACKEND_Input;
typedef struct TRITONBACKEND_Output TRITONBACKEND_Output;
typedef struct TRITONBACKEND_State TRITONBACKEND_State;
typedef struct TRITONBACKEND_Request TRITONBACKEND_Request;
typedef struct TRITONBACKEND_ResponseFactory TRITONBACKEND_ResponseFactory;
typedef struct TRITONBACKEND_Response TRITONBACKEND_Response;
typedef struct TRITONBACKEND_Backend TRITONBACKEND_Backend;
typedef struct TRITONBACKEND_Model TRITONBACKEND_Model;
typedef struct TRITONBACKEND_ModelInstance TRITONBACKEND_ModelInstance;
typedef struct TRITONBACKEND_ModelInstanceResponseStatistics TRITONBACKEND_ModelInstanceResponseStatistics;
typedef struct TRITONBACKEND_BackendAttribute TRITONBACKEND_BackendAttribute;
typedef struct TRITONBACKEND_Batcher TRITONBACKEND_Batcher;
#include <triton/core/tritonbackend.h>
// Literal callback typedefs avoid a pinned translate-c/debug-info issue:
// typeof(function) becomes Zig @TypeOf(function), which emits relocations to
// plugin symbols even in resolve-only clients. C checks every alias against
// the authoritative upstream declaration so signature drift is a build error.
typedef TRITONSERVER_Error* TRITONZIG_BackendInitializeFn(TRITONBACKEND_Backend*);
typedef TRITONZIG_BackendInitializeFn TRITONZIG_BackendFinalizeFn;
typedef TRITONSERVER_Error* TRITONZIG_ModelInitializeFn(TRITONBACKEND_Model*);
typedef TRITONZIG_ModelInitializeFn TRITONZIG_ModelFinalizeFn;
typedef TRITONSERVER_Error* TRITONZIG_ModelInstanceInitializeFn(TRITONBACKEND_ModelInstance*);
typedef TRITONZIG_ModelInstanceInitializeFn TRITONZIG_ModelInstanceFinalizeFn;
typedef TRITONZIG_ModelInstanceInitializeFn TRITONZIG_ModelInstanceReadyFn;
typedef TRITONSERVER_Error* TRITONZIG_ModelInstanceExecuteFn(TRITONBACKEND_ModelInstance*, TRITONBACKEND_Request**, uint32_t);
typedef TRITONSERVER_Error* TRITONZIG_GetBackendAttributeFn(TRITONBACKEND_Backend*, TRITONBACKEND_BackendAttribute*);
#define TRITONZIG_VERIFY_CALLBACK(alias, symbol) \
    enum { TRITONZIG_ABI_CHECK_##symbol = __builtin_types_compatible_p(alias, __typeof__(symbol)) }; \
    _Static_assert(TRITONZIG_ABI_CHECK_##symbol, "backend callback ABI mismatch: " #symbol)
TRITONZIG_VERIFY_CALLBACK(TRITONZIG_BackendInitializeFn, TRITONBACKEND_Initialize);
TRITONZIG_VERIFY_CALLBACK(TRITONZIG_BackendFinalizeFn, TRITONBACKEND_Finalize);
TRITONZIG_VERIFY_CALLBACK(TRITONZIG_ModelInitializeFn, TRITONBACKEND_ModelInitialize);
TRITONZIG_VERIFY_CALLBACK(TRITONZIG_ModelFinalizeFn, TRITONBACKEND_ModelFinalize);
TRITONZIG_VERIFY_CALLBACK(TRITONZIG_ModelInstanceInitializeFn, TRITONBACKEND_ModelInstanceInitialize);
TRITONZIG_VERIFY_CALLBACK(TRITONZIG_ModelInstanceFinalizeFn, TRITONBACKEND_ModelInstanceFinalize);
TRITONZIG_VERIFY_CALLBACK(TRITONZIG_ModelInstanceExecuteFn, TRITONBACKEND_ModelInstanceExecute);
TRITONZIG_VERIFY_CALLBACK(TRITONZIG_ModelInstanceReadyFn, TRITONBACKEND_ModelInstanceReady);
TRITONZIG_VERIFY_CALLBACK(TRITONZIG_GetBackendAttributeFn, TRITONBACKEND_GetBackendAttribute);
#undef TRITONZIG_VERIFY_CALLBACK
#endif
