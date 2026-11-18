#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static _Atomic unsigned long triton_graph_launches;

/* CUDA's Linux ABI passes graph and stream handles as pointers and returns
 * the cudaError_t status in an integer register. The benchmark only uses
 * this interposer in untimed runs. */
int cudaGraphLaunch(void *graph, void *stream) {
    typedef int (*Launch)(void *, void *);
    Launch launch = (Launch)dlsym(RTLD_NEXT, "cudaGraphLaunch");
    if (launch == NULL) {
        fputs("could not resolve the CUDA runtime's cudaGraphLaunch\n", stderr);
        abort();
    }
    Dl_info caller;
    int from_backend = dladdr(__builtin_return_address(0), &caller) &&
                       strstr(caller.dli_fname, "/libtriton_tensorrt.so") != NULL;
    int status = launch(graph, stream);
    if (from_backend && status == 0)
        atomic_fetch_add_explicit(&triton_graph_launches, 1, memory_order_relaxed);
    return status;
}

__attribute__((destructor))
static void report_launches(void) {
    fprintf(stderr, "TensorRT backend cudaGraphLaunch successes: %lu\n",
            atomic_load_explicit(&triton_graph_launches, memory_order_relaxed));
}
