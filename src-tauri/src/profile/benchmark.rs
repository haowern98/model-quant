use std::path::PathBuf;
use std::time::Instant;

use crate::progress::{ProgressEmitter, ProgressStage};

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchmarkResult {
    pub prompt_eval_tps: f64,
    pub token_gen_tps: f64,
    pub ttft_ms: f64,
    pub vram_peak_mb: f64,
    pub vram_allocated_mb: f64,
    pub disk_size_mb: f64,
    pub elapsed_ms: f64,
    pub load_ms: f64,
    pub test_mode: String,
    pub status_message: String,
    pub native_runtime: Option<String>,
    pub model_tensor_count: Option<u64>,
    pub model_metadata_count: Option<u64>,
}

pub fn run_benchmark(
    gguf_path: &PathBuf,
    _prompt_tokens: u32,
    progress: &ProgressEmitter,
) -> Result<BenchmarkResult, String> {
    let start = Instant::now();

    let disk_size = std::fs::metadata(gguf_path)
        .map(|m| m.len() as f64 / (1024.0 * 1024.0))
        .unwrap_or(0.0);

    progress.loading(0.3);

    // v1 benchmark: file size + elapsed time measurement.
    // Full inference requires llama.cpp linked — this stub covers the
    // progress reporting pipeline and can be upgraded once FFI is linked.

    progress.loading(0.7);

    // Measure VRAM via C++ profiler
    unsafe { crate::ffi::profiler_bindings::profiler_reset_peak(); }
    let mut vram_allocated_mb = 0.0f64;
    let mut vram_peak_mb = 0.0f64;
    unsafe {
        crate::ffi::profiler_bindings::profiler_get_vram(
            &mut vram_allocated_mb,
            &mut vram_peak_mb,
        );
    }

    progress.benchmarking(0.5);

    let elapsed = start.elapsed();

    Ok(BenchmarkResult {
        prompt_eval_tps: 0.0,
        token_gen_tps: 0.0,
        ttft_ms: 0.0,
        vram_peak_mb,
        vram_allocated_mb,
        disk_size_mb: disk_size,
        elapsed_ms: elapsed.as_millis() as f64,
        load_ms: 0.0,
        test_mode: "file_size_stub".to_string(),
        status_message: "Legacy benchmark stub measured file size only".to_string(),
        native_runtime: None,
        model_tensor_count: None,
        model_metadata_count: None,
    })
}

pub fn run_native_runtime_smoke(
    gguf_path: &PathBuf,
    progress: &ProgressEmitter,
) -> Result<BenchmarkResult, String> {
    let start = Instant::now();

    let disk_size = std::fs::metadata(gguf_path)
        .map(|m| m.len() as f64 / (1024.0 * 1024.0))
        .unwrap_or(0.0);

    progress.emit(ProgressStage::Loading, 0.25, "Inspecting GGUF with native runtime...");
    let summary = crate::ffi::runtime_bindings::inspect_gguf(&gguf_path.to_string_lossy())?;
    progress.emit(ProgressStage::Loading, 1.0, "Native runtime inspection complete");

    unsafe { crate::ffi::profiler_bindings::profiler_reset_peak(); }
    let mut vram_allocated_mb = 0.0f64;
    let mut vram_peak_mb = 0.0f64;
    unsafe {
        crate::ffi::profiler_bindings::profiler_get_vram(
            &mut vram_allocated_mb,
            &mut vram_peak_mb,
        );
    }

    progress.emit(ProgressStage::Benchmarking, 1.0, "Native runtime smoke test complete");

    let elapsed = start.elapsed();
    let runtime = crate::ffi::runtime_bindings::runtime_version();
    let system_info = crate::ffi::runtime_bindings::llama_system_info();

    Ok(BenchmarkResult {
        prompt_eval_tps: 0.0,
        token_gen_tps: 0.0,
        ttft_ms: 0.0,
        vram_peak_mb,
        vram_allocated_mb,
        disk_size_mb: disk_size,
        elapsed_ms: elapsed.as_millis() as f64,
        load_ms: 0.0,
        test_mode: "native_runtime_smoke".to_string(),
        status_message: format!(
            "Native runtime inspected GGUF v{} with {} tensors. Recipe execution is not implemented yet.",
            summary.version, summary.tensor_count
        ),
        native_runtime: Some(format!("{} | {}", runtime, system_info.trim())),
        model_tensor_count: Some(summary.tensor_count),
        model_metadata_count: Some(summary.metadata_count),
    })
}

pub fn run_native_baseline_benchmark(
    gguf_path: &PathBuf,
    max_tokens: u32,
    progress: &ProgressEmitter,
) -> Result<BenchmarkResult, String> {
    let start = Instant::now();

    let disk_size = std::fs::metadata(gguf_path)
        .map(|m| m.len() as f64 / (1024.0 * 1024.0))
        .unwrap_or(0.0);

    let max_tokens = max_tokens.clamp(1, 16);
    let prompt = "The capital of France is";

    progress.emit(ProgressStage::Loading, 0.1, "Loading GGUF with native llama.cpp...");
    let summary = crate::ffi::runtime_bindings::inspect_gguf(&gguf_path.to_string_lossy())?;
    let benchmark = crate::ffi::runtime_bindings::benchmark_baseline(
        &gguf_path.to_string_lossy(),
        prompt,
        max_tokens,
    )?;
    progress.emit(ProgressStage::Benchmarking, 1.0, "Native baseline inference complete");

    unsafe { crate::ffi::profiler_bindings::profiler_reset_peak(); }
    let mut vram_allocated_mb = 0.0f64;
    let mut vram_peak_mb = 0.0f64;
    unsafe {
        crate::ffi::profiler_bindings::profiler_get_vram(
            &mut vram_allocated_mb,
            &mut vram_peak_mb,
        );
    }

    let elapsed = start.elapsed();
    let runtime = crate::ffi::runtime_bindings::runtime_version();
    let system_info = crate::ffi::runtime_bindings::llama_system_info();

    Ok(BenchmarkResult {
        prompt_eval_tps: benchmark.prompt_eval_tps,
        token_gen_tps: benchmark.token_gen_tps,
        ttft_ms: benchmark.ttft_ms,
        vram_peak_mb,
        vram_allocated_mb,
        disk_size_mb: disk_size,
        elapsed_ms: elapsed.as_millis() as f64,
        load_ms: benchmark.load_ms,
        test_mode: "native_baseline".to_string(),
        status_message: format!(
            "Native llama.cpp baseline loaded GGUF v{} with {} tensors, evaluated {} prompt tokens, and generated {} tokens. Recipe quant overrides are not active for this run.",
            summary.version, summary.tensor_count, benchmark.prompt_tokens, benchmark.generated_tokens
        ),
        native_runtime: Some(format!("{} | {}", runtime, system_info.trim())),
        model_tensor_count: Some(summary.tensor_count),
        model_metadata_count: Some(summary.metadata_count),
    })
}
