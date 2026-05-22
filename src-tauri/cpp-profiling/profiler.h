#pragma once

#include <cstddef>
#include <cstdint>

extern "C" {

void profiler_init();
void profiler_shutdown();
void profiler_get_vram(double* allocated_mb, double* peak_mb);
void profiler_timer_start(const char* name);
int64_t profiler_timer_stop(const char* name);
void profiler_reset_peak();

} // extern "C"
