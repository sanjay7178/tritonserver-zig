# Zig batching proof fixture

This test-only CPU backend echoes INT32 `IN0` as `OUT0` and returns optional
INT64 `BATCH_INFO` rows `[execution_id, total_samples, execute_request_count]`.
Each logical sample in a combined request receives the same execution ID and
counts. Its model configurations omit native `dynamic_batching` deliberately.
A batch size greater than one with execute-request-count one therefore proves
that Zig submitted one concatenated request rather than asking Triton to gather
independent requests.

`phase4_multi` supplies two independent INT32 inputs, `IN0` and `IN1`, echoed as
`OUT0` and `OUT1`. This checks that the scheduler concatenates each named input
and splits every output while canonicalizing caller input and output ordering.

The backend honors requested output names. A first INT32 value equal to
`INT32_MIN` causes an asynchronous InvalidArg response. A first value equal to
`INT32_MAX` keeps execution busy for one second for bounded-queue tests; benchmark
inputs avoid this sentinel. The model named
`phase4_malformed` deliberately produces an OUT0 leading dimension one larger
than its input so scheduler response-splitting validation can be exercised.
Neither negative behavior is part of the licensed minimal backend.
