cmake_minimum_required(VERSION 3.20)

project(model_surgery_native LANGUAGES C CXX)

set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_CUDA ON CACHE BOOL "" FORCE)
set(GGML_CUDA_GRAPHS ON CACHE BOOL "" FORCE)
set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architecture for RTX 40-series Ada GPUs" FORCE)

add_subdirectory(llama.cpp EXCLUDE_FROM_ALL)
# mtmd normally inherits this from llama.cpp's tools directory. We add the
# library directly so the runtime can link it without enabling every tool.
set(LLAMA_INSTALL_VERSION 0.0.0)
add_subdirectory(llama.cpp/tools/mtmd EXCLUDE_FROM_ALL)
add_subdirectory(model_surgery_runtime)
