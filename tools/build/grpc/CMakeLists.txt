cmake_minimum_required(VERSION 3.31.8)
project(tritonzig_grpc LANGUAGES C CXX)

set(TRITONZIG_TRITON_PREFIX "" CACHE PATH "Installation of the real Triton core")
set(TRITONZIG_SERVER_SOURCE "" CACHE PATH "Pinned upstream Triton server source")
set(TRITONZIG_COMMON_SOURCE "" CACHE PATH "Pinned upstream Triton common source")

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "The gRPC frontend build requires Linux")
endif()
if(NOT EXISTS "${TRITONZIG_TRITON_PREFIX}/include/triton/core/tritonserver.h"
    OR NOT EXISTS "${TRITONZIG_TRITON_PREFIX}/lib/libtritonserver.so")
  message(FATAL_ERROR "TRITONZIG_TRITON_PREFIX must contain Triton headers and lib/libtritonserver.so")
endif()
if(NOT EXISTS "${TRITONZIG_SERVER_SOURCE}/src/grpc/CMakeLists.txt"
    OR NOT EXISTS "${TRITONZIG_COMMON_SOURCE}/protobuf/grpc_service.proto")
  message(FATAL_ERROR "Provide the pinned Triton server and common source directories")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_SHARED_LIBS OFF)
set(TRITON_MIN_CXX_STANDARD 17)
set(TRITON_COMMON_ENABLE_PROTOBUF ON)
set(TRITON_COMMON_ENABLE_PROTOBUF_PYTHON OFF)
set(TRITON_COMMON_ENABLE_GRPC ON)
set(TRITON_ENABLE_LOGGING ON)
set(TRITON_ENABLE_STATS ON)
set(TRITON_ENABLE_TRACING OFF)
# GPU inference for inline tensors runs in the selected Triton core.
# This frontend leaves CUDA shared-memory registration disabled.
set(TRITON_ENABLE_GPU OFF)

# Load the same native Protobuf/GRPC toolchain in both upstream subdirectories.
set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)
find_path(B64_INCLUDE_DIR b64/cdecode.h REQUIRED)
find_library(B64_LIBRARY NAMES b64 REQUIRED)

add_library(triton-core-serverapi INTERFACE)
target_include_directories(triton-core-serverapi INTERFACE "${TRITONZIG_TRITON_PREFIX}/include")
# Satisfy the upstream link target with the actual core used by Zig.
add_library(triton-core-serverstub SHARED IMPORTED GLOBAL)
set_target_properties(triton-core-serverstub PROPERTIES
  IMPORTED_LOCATION "${TRITONZIG_TRITON_PREFIX}/lib/libtritonserver.so")

add_subdirectory("${TRITONZIG_COMMON_SOURCE}" common EXCLUDE_FROM_ALL)
set(upstream_src "${TRITONZIG_SERVER_SOURCE}/src")
add_subdirectory("${upstream_src}/grpc" grpc EXCLUDE_FROM_ALL)

# The upstream tritonfrontend bindings use these same supporting sources.
add_library(tritonzig_grpc SHARED
  bridge.cc
  "${upstream_src}/common.cc"
  "${upstream_src}/classification.cc"
  "${upstream_src}/shared_memory_manager.cc")
target_compile_features(tritonzig_grpc PRIVATE cxx_std_17)
target_compile_options(tritonzig_grpc PRIVATE -Wall -Wextra)
target_include_directories(tritonzig_grpc PRIVATE "${upstream_src}" "${B64_INCLUDE_DIR}")
target_link_libraries(tritonzig_grpc PRIVATE grpc-endpoint-library "${B64_LIBRARY}")
target_link_options(tritonzig_grpc PRIVATE
  "LINKER:-z,defs"
  "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
set_target_properties(tritonzig_grpc PROPERTIES
  BUILD_RPATH "${TRITONZIG_TRITON_PREFIX}/lib"
  INSTALL_RPATH "${TRITONZIG_TRITON_PREFIX}/lib"
  LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
install(TARGETS tritonzig_grpc LIBRARY DESTINATION lib)
