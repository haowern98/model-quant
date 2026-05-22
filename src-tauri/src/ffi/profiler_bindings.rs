extern "C" {
    pub fn profiler_init();
    pub fn profiler_shutdown();
    pub fn profiler_get_vram(allocated_mb: *mut f64, peak_mb: *mut f64);
    pub fn profiler_timer_start(name: *const std::os::raw::c_char);
    pub fn profiler_timer_stop(name: *const std::os::raw::c_char) -> i64;
    pub fn profiler_reset_peak();
}
