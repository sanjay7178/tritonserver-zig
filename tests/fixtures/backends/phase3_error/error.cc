// Deliberately failing CPU backend used only to test asynchronous error delivery.
// Returning an execute error asks Triton to create an error response and return
// request ownership. No input/output buffers are accessed by this fixture.
#include "triton/core/tritonbackend.h"

extern "C" TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance*, TRITONBACKEND_Request**, const uint32_t)
{
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_INVALID_ARG, "intentional phase3 test backend failure");
}
