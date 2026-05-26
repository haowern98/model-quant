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
    pub copied_tensor_count: u64,
    pub converted_tensor_count: u64,
    pub converted_bytes_before: u64,
    pub converted_bytes_after: u64,
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
    unsafe {
        crate::ffi::profiler_bindings::profiler_reset_peak();
    }
    let mut vram_allocated_mb = 0.0f64;
    let mut vram_peak_mb = 0.0f64;
    unsafe {
        crate::ffi::profiler_bindings::profiler_get_vram(&mut vram_allocated_mb, &mut vram_peak_mb);
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
        copied_tensor_count: 0,
        converted_tensor_count: 0,
        converted_bytes_before: 0,
        converted_bytes_after: 0,
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

    progress.emit(
        ProgressStage::Loading,
        0.25,
        "Inspecting GGUF with native runtime...",
    );
    let summary = crate::ffi::runtime_bindings::inspect_gguf(&gguf_path.to_string_lossy())?;
    progress.emit(
        ProgressStage::Loading,
        1.0,
        "Native runtime inspection complete",
    );

    unsafe {
        crate::ffi::profiler_bindings::profiler_reset_peak();
    }
    let mut vram_allocated_mb = 0.0f64;
    let mut vram_peak_mb = 0.0f64;
    unsafe {
        crate::ffi::profiler_bindings::profiler_get_vram(&mut vram_allocated_mb, &mut vram_peak_mb);
    }

    progress.emit(
        ProgressStage::Benchmarking,
        1.0,
        "Native runtime smoke test complete",
    );

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
        copied_tensor_count: 0,
        converted_tensor_count: 0,
        converted_bytes_before: 0,
        converted_bytes_after: 0,
    })
}

pub fn run_native_baseline_benchmark(
    gguf_path: &PathBuf,
    max_tokens: u32,
    progress: &ProgressEmitter,
) -> Result<BenchmarkResult, String> {
    run_native_inference_benchmark(
        gguf_path,
        max_tokens,
        progress,
        "Loading GGUF with native llama.cpp...",
        "Native baseline inference complete",
        "native_baseline",
        |path, prompt, max_tokens| {
            crate::ffi::runtime_bindings::benchmark_baseline(path, prompt, max_tokens)
        },
        |summary, benchmark| {
            format!(
                "Native llama.cpp baseline loaded GGUF v{} with {} tensors, evaluated {} prompt tokens, and generated {} tokens. Recipe quant overrides are not active for this run.",
                summary.version, summary.tensor_count, benchmark.prompt_tokens, benchmark.generated_tokens
            )
        },
    )
}

pub fn run_native_user_copy_benchmark(
    gguf_path: &PathBuf,
    max_tokens: u32,
    progress: &ProgressEmitter,
) -> Result<BenchmarkResult, String> {
    run_native_inference_benchmark(
        gguf_path,
        max_tokens,
        progress,
        "Loading GGUF through native user-model path...",
        "Native user-model inference complete",
        "native_user_copy",
        |path, prompt, max_tokens| {
            crate::ffi::runtime_bindings::benchmark_user_copy(path, prompt, max_tokens)
        },
        |summary, benchmark| {
            format!(
                "Native llama.cpp user-model path copied GGUF v{} with {} tensors into backend buffers, evaluated {} prompt tokens, and generated {} tokens. Changed tensor conversion is not active for this run.",
                summary.version, summary.tensor_count, benchmark.prompt_tokens, benchmark.generated_tokens
            )
        },
    )
}

pub fn run_native_recipe_benchmark(
    gguf_path: &PathBuf,
    targets: &[(String, String)],
    max_tokens: u32,
    progress: &ProgressEmitter,
) -> Result<BenchmarkResult, String> {
    run_native_inference_benchmark(
        gguf_path,
        max_tokens,
        progress,
        "Loading GGUF through native recipe path...",
        "Native recipe inference complete",
        "native_recipe_phase1",
        |path, prompt, max_tokens| {
            crate::ffi::runtime_bindings::benchmark_recipe(path, targets, prompt, max_tokens)
        },
        |summary, benchmark| {
            format!(
                "Native llama.cpp recipe path validated {} tensor target(s), copied unchanged tensors and applied supported in-memory conversions from GGUF v{}, evaluated {} prompt tokens, and generated {} tokens.",
                targets.len(), summary.version, benchmark.prompt_tokens, benchmark.generated_tokens
            )
        },
    )
}

fn run_native_inference_benchmark(
    gguf_path: &PathBuf,
    max_tokens: u32,
    progress: &ProgressEmitter,
    loading_message: &str,
    complete_message: &str,
    test_mode: &str,
    run_benchmark: impl FnOnce(
        &str,
        &str,
        u32,
    ) -> Result<crate::ffi::runtime_bindings::MsBaselineBenchmark, String>,
    status_message: impl FnOnce(
        &crate::ffi::runtime_bindings::MsGgufSummary,
        &crate::ffi::runtime_bindings::MsBaselineBenchmark,
    ) -> String,
) -> Result<BenchmarkResult, String> {
    let start = Instant::now();

    let disk_size = std::fs::metadata(gguf_path)
        .map(|m| m.len() as f64 / (1024.0 * 1024.0))
        .unwrap_or(0.0);

    let max_tokens = max_tokens.clamp(1, 16);
    let prompt = "The capital of France is";

    progress.emit(ProgressStage::Loading, 0.1, loading_message);
    let summary = crate::ffi::runtime_bindings::inspect_gguf(&gguf_path.to_string_lossy())?;
    let benchmark = run_benchmark(&gguf_path.to_string_lossy(), prompt, max_tokens)?;
    progress.emit(ProgressStage::Benchmarking, 1.0, complete_message);

    let elapsed = start.elapsed();
    let runtime = crate::ffi::runtime_bindings::runtime_version();
    let system_info = crate::ffi::runtime_bindings::llama_system_info();

    Ok(BenchmarkResult {
        prompt_eval_tps: benchmark.prompt_eval_tps,
        token_gen_tps: benchmark.token_gen_tps,
        ttft_ms: benchmark.ttft_ms,
        vram_peak_mb: benchmark.vram_peak_mb,
        vram_allocated_mb: benchmark.vram_allocated_mb,
        disk_size_mb: disk_size,
        elapsed_ms: elapsed.as_millis() as f64,
        load_ms: benchmark.load_ms,
        test_mode: test_mode.to_string(),
        status_message: status_message(&summary, &benchmark),
        native_runtime: Some(format!("{} | {}", runtime, system_info.trim())),
        model_tensor_count: Some(summary.tensor_count),
        model_metadata_count: Some(summary.metadata_count),
        copied_tensor_count: benchmark.copied_tensor_count,
        converted_tensor_count: benchmark.converted_tensor_count,
        converted_bytes_before: benchmark.converted_bytes_before,
        converted_bytes_after: benchmark.converted_bytes_after,
    })
}
