#include "profiler.h"

#include <cuda_runtime.h>
#include <chrono>
#include <string>
#include <unordered_map>

static std::unordered_map<std::string, std::chrono::steady_clock::time_point> timers;
static double peak_vram_mb = 0.0;

void profiler_init() {
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        return;
    }
    cudaFree(0);
    cudaDeviceSynchronize();
}

void profiler_shutdown() {
    timers.clear();
    cudaDeviceReset();
}

void profiler_get_vram(double* allocated_mb, double* peak_mb) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;

    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess) {
        *allocated_mb = 0.0;
        *peak_mb = 0.0;
        return;
    }

    double used_mb = (total_bytes - free_bytes) / (1024.0 * 1024.0);
    *allocated_mb = used_mb;

    if (used_mb > peak_vram_mb) {
        peak_vram_mb = used_mb;
    }
    *peak_mb = peak_vram_mb;
}

void profiler_timer_start(const char* name) {
    timers[std::string(name)] = std::chrono::steady_clock::now();
}

int64_t profiler_timer_stop(const char* name) {
    auto it = timers.find(std::string(name));
    if (it == timers.end()) {
        return -1;
    }
    auto elapsed = std::chrono::steady_clock::now() - it->second;
    timers.erase(it);
    return std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
}

void profiler_reset_peak() {
    peak_vram_mb = 0.0;
}
