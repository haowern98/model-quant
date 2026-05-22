// Stub profiler — compiled when CUDA toolkit is not available.
// Returns zero for all measurements. The app functions normally,
// just without GPU profiling metrics.

#include "profiler.h"
#include <cstdint>

void profiler_init() {}
void profiler_shutdown() {}

void profiler_get_vram(double* allocated_mb, double* peak_mb) {
    *allocated_mb = 0.0;
    *peak_mb = 0.0;
}

void profiler_timer_start(const char*) {}
int64_t profiler_timer_stop(const char*) { return 0; }
void profiler_reset_peak() {}
