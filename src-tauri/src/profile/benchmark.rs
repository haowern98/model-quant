use std::path::PathBuf;
use std::time::Instant;

use crate::progress::{ProgressEmitter, ProgressStage};

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchmarkResult {
    pub prompt_eval_tps: f64,
    pub token_gen_tps: f64,
    pub ttft_ms: f64,
    pub prompt_eval_ms: f64,
    pub generation_ms: f64,
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
    pub requested_target_count: u64,
    pub verified_target_count: u64,
    pub baseline_benchmark: Option<RuntimeBenchmark>,
    pub quality_eval: Option<RecipeQualityEval>,
    pub standard_eval: Option<StandardEvalReport>,
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RecipeQualityEval {
    pub baseline_nll: Option<f64>,
    pub baseline_ppl: Option<f64>,
    pub baseline_ppl_uncertainty: Option<f64>,
    pub baseline_eval_ms: Option<f64>,
    pub baseline_vram_peak_mb: Option<f64>,
    pub baseline_vram_allocated_mb: Option<f64>,
    pub recipe_nll: f64,
    pub recipe_ppl: f64,
    pub recipe_ppl_uncertainty: f64,
    pub recipe_eval_ms: f64,
    pub recipe_vram_peak_mb: f64,
    pub recipe_vram_allocated_mb: f64,
    pub ppl_delta: f64,
    pub ppl_delta_percent: f64,
    pub eval_token_count: u64,
    pub eval_sample_count: u64,
    pub skipped_sample_count: u64,
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeBenchmark {
    pub prompt_eval_tps: f64,
    pub token_gen_tps: f64,
    pub ttft_ms: f64,
    pub prompt_eval_ms: f64,
    pub generation_ms: f64,
    pub vram_peak_mb: f64,
    pub vram_allocated_mb: f64,
    pub load_ms: f64,
    pub elapsed_ms: f64,
    pub model_tensor_count: Option<u64>,
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StandardEvalReport {
    pub sample_count: u64,
    pub task_count: u64,
    pub baseline_accuracy: Option<f64>,
    pub recipe_accuracy: f64,
    pub accuracy_delta: Option<f64>,
    pub correct_to_wrong_count: u64,
    pub wrong_to_correct_count: u64,
    pub baseline_avg_margin: Option<f64>,
    pub recipe_avg_margin: f64,
    pub margin_delta: Option<f64>,
    pub tasks: Vec<StandardEvalTaskReport>,
    pub sample_audits: Vec<StandardEvalSampleAuditReport>,
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StandardEvalTaskReport {
    pub task: String,
    pub sample_count: u64,
    pub baseline_correct_count: Option<u64>,
    pub recipe_correct_count: u64,
    pub correct_to_wrong_count: u64,
    pub wrong_to_correct_count: u64,
    pub same_prediction_count: u64,
    pub baseline_accuracy: Option<f64>,
    pub recipe_accuracy: f64,
    pub accuracy_delta: Option<f64>,
    pub baseline_avg_margin: Option<f64>,
    pub recipe_avg_margin: f64,
    pub margin_delta: Option<f64>,
    pub baseline_avg_correct_nll: Option<f64>,
    pub recipe_avg_correct_nll: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StandardEvalSampleAuditReport {
    pub task: String,
    pub doc_id: String,
    pub sample_index: u64,
    pub prompt: String,
    pub target_delimiter: String,
    pub gold_index: u32,
    pub baseline_prediction_index: Option<u32>,
    pub recipe_prediction_index: u32,
    pub baseline_correct: Option<bool>,
    pub recipe_correct: bool,
    pub flip_type: String,
    pub choices: Vec<StandardEvalChoiceAuditReport>,
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StandardEvalChoiceAuditReport {
    pub index: u32,
    pub choice: String,
    pub continuation: String,
    pub denominator: f64,
    pub baseline_nll: Option<f64>,
    pub baseline_loglikelihood: Option<f64>,
    pub baseline_score: Option<f64>,
    pub recipe_nll: f64,
    pub recipe_loglikelihood: f64,
    pub recipe_score: f64,
}

#[derive(Debug, serde::Deserialize)]
struct EvalText {
    text: String,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct StandardSubset {
    ppl: Vec<EvalText>,
    tasks: Vec<StandardTask>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct StandardTask {
    name: String,
    output_type: String,
    samples: Vec<StandardSample>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct StandardSample {
    prompt: String,
    target_delimiter: Option<String>,
    choices: Vec<String>,
    gold: u32,
    normalize_by_choice_length: bool,
    doc_id: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StandardEvalPreset {
    Quick,
    Default,
}

impl StandardEvalPreset {
    fn label(self) -> &'static str {
        match self {
            StandardEvalPreset::Quick => "quick",
            StandardEvalPreset::Default => "default",
        }
    }
}

pub fn parse_standard_eval_preset(value: Option<&str>) -> Result<StandardEvalPreset, String> {
    match value.unwrap_or("default") {
        "quick" => Ok(StandardEvalPreset::Quick),
        "default" => Ok(StandardEvalPreset::Default),
        preset => Err(format!("Unknown eval preset: {}", preset)),
    }
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
        prompt_eval_ms: 0.0,
        generation_ms: 0.0,
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
        requested_target_count: 0,
        verified_target_count: 0,
        baseline_benchmark: None,
        quality_eval: None,
        standard_eval: None,
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
        prompt_eval_ms: 0.0,
        generation_ms: 0.0,
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
        requested_target_count: 0,
        verified_target_count: 0,
        baseline_benchmark: None,
        quality_eval: None,
        standard_eval: None,
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
        None,
        |path, prompt, max_tokens| {
            crate::ffi::runtime_bindings::benchmark_baseline(path, prompt, max_tokens)
                .map(benchmark_only)
        },
        |summary, benchmark, _, _| {
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
        None,
        |path, prompt, max_tokens| {
            crate::ffi::runtime_bindings::benchmark_user_copy(path, prompt, max_tokens)
                .map(benchmark_only)
        },
        |summary, benchmark, _, _| {
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
    run_native_recipe_compare_benchmark(
        gguf_path,
        targets,
        max_tokens,
        StandardEvalPreset::Default,
        progress,
    )
}

pub fn run_native_recipe_single_benchmark(
    gguf_path: &PathBuf,
    targets: &[(String, String)],
    max_tokens: u32,
    eval_preset: StandardEvalPreset,
    progress: &ProgressEmitter,
) -> Result<BenchmarkResult, String> {
    let standard_subset = load_standard_eval_subset(eval_preset)?;
    let eval_texts = standard_subset.ppl_texts;
    let standard_samples = standard_subset.samples;
    run_native_inference_benchmark(
        gguf_path,
        max_tokens,
        progress,
        "Running single recipe model test...",
        "Native single recipe test complete",
        "native_recipe_single_v1",
        Some(&standard_samples),
        |path, prompt, max_tokens| {
            crate::ffi::runtime_bindings::eval_recipe_standard_single(
                path,
                targets,
                &eval_texts,
                &standard_samples,
                128,
                prompt,
                max_tokens,
            )
            .map(|(benchmark, eval, standard, audits)| {
                (benchmark, Some(eval), Some(standard), Some(audits))
            })
        },
        |summary, benchmark, eval, standard| {
            format!(
                "Native llama.cpp recipe path validated {} tensor target(s), ran {} built-in lm-eval-style local eval with {} llama.cpp PPL tokens and {} frozen standard task sample(s) from GGUF v{}, copied unchanged tensors and applied supported in-memory conversions, then generated {} tokens.",
                targets.len(),
                eval_preset.label(),
                eval.map(|quality| quality.eval_token_count).unwrap_or(0),
                standard.map(|report| report.sample_count).unwrap_or(0),
                summary.version,
                benchmark.generated_tokens
            )
        },
    )
}

pub fn run_native_recipe_compare_benchmark(
    gguf_path: &PathBuf,
    targets: &[(String, String)],
    max_tokens: u32,
    eval_preset: StandardEvalPreset,
    progress: &ProgressEmitter,
) -> Result<BenchmarkResult, String> {
    let standard_subset = load_standard_eval_subset(eval_preset)?;
    let eval_texts = standard_subset.ppl_texts;
    let standard_samples = standard_subset.samples;
    run_native_inference_benchmark(
        gguf_path,
        max_tokens,
        progress,
        "Running baseline and recipe drift eval...",
        "Native recipe eval complete",
        "native_recipe_eval_v1",
        Some(&standard_samples),
        |path, prompt, max_tokens| {
            crate::ffi::runtime_bindings::eval_recipe_standard(
                path,
                targets,
                &eval_texts,
                &standard_samples,
                128,
                prompt,
                max_tokens,
            )
            .map(|(benchmark, eval, standard, audits)| {
                (benchmark, Some(eval), Some(standard), Some(audits))
            })
        },
        |summary, benchmark, eval, standard| {
            format!(
                "Native llama.cpp recipe path validated {} tensor target(s), ran {} built-in lm-eval-style recipe drift eval with {} llama.cpp PPL tokens and {} frozen standard task sample(s) from GGUF v{}, copied unchanged tensors and applied supported in-memory conversions, then generated {} tokens.",
                targets.len(),
                eval_preset.label(),
                eval.map(|quality| quality.eval_token_count).unwrap_or(0),
                standard.map(|report| report.sample_count).unwrap_or(0),
                summary.version,
                benchmark.generated_tokens
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
    standard_samples_for_report: Option<&[crate::ffi::runtime_bindings::StandardEvalSampleInput]>,
    run_benchmark: impl FnOnce(
        &str,
        &str,
        u32,
    ) -> Result<
        (
            crate::ffi::runtime_bindings::MsBaselineBenchmark,
            Option<crate::ffi::runtime_bindings::MsRecipeEvalResult>,
            Option<Vec<crate::ffi::runtime_bindings::MsStandardEvalTaskResult>>,
            Option<Vec<crate::ffi::runtime_bindings::MsStandardEvalSampleAudit>>,
        ),
        String,
    >,
    status_message: impl FnOnce(
        &crate::ffi::runtime_bindings::MsGgufSummary,
        &crate::ffi::runtime_bindings::MsBaselineBenchmark,
        Option<&RecipeQualityEval>,
        Option<&StandardEvalReport>,
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
    let (benchmark, eval, standard_tasks, standard_sample_audits) =
        run_benchmark(&gguf_path.to_string_lossy(), prompt, max_tokens)?;
    let baseline_benchmark = eval.as_ref().and_then(|native| {
        baseline_runtime_benchmark_from_native(native, Some(summary.tensor_count))
    });
    let quality_eval = eval.map(recipe_quality_eval_from_native);
    let standard_eval = standard_tasks.as_deref().map(|tasks| {
        standard_eval_report_from_native(
            tasks,
            standard_sample_audits.as_deref().unwrap_or(&[]),
            standard_samples_for_report.unwrap_or(&[]),
            quality_eval.as_ref(),
        )
    });
    progress.emit(ProgressStage::Benchmarking, 1.0, complete_message);

    let elapsed = start.elapsed();
    let runtime = crate::ffi::runtime_bindings::runtime_version();
    let system_info = crate::ffi::runtime_bindings::llama_system_info();

    Ok(BenchmarkResult {
        prompt_eval_tps: benchmark.prompt_eval_tps,
        token_gen_tps: benchmark.token_gen_tps,
        ttft_ms: benchmark.ttft_ms,
        prompt_eval_ms: benchmark.prompt_eval_ms,
        generation_ms: benchmark.generation_ms,
        vram_peak_mb: benchmark.vram_peak_mb,
        vram_allocated_mb: benchmark.vram_allocated_mb,
        disk_size_mb: disk_size,
        elapsed_ms: elapsed.as_millis() as f64,
        load_ms: benchmark.load_ms,
        test_mode: test_mode.to_string(),
        status_message: status_message(
            &summary,
            &benchmark,
            quality_eval.as_ref(),
            standard_eval.as_ref(),
        ),
        native_runtime: Some(format!("{} | {}", runtime, system_info.trim())),
        model_tensor_count: Some(summary.tensor_count),
        model_metadata_count: Some(summary.metadata_count),
        copied_tensor_count: benchmark.copied_tensor_count,
        converted_tensor_count: benchmark.converted_tensor_count,
        converted_bytes_before: benchmark.converted_bytes_before,
        converted_bytes_after: benchmark.converted_bytes_after,
        requested_target_count: benchmark.requested_target_count,
        verified_target_count: benchmark.verified_target_count,
        baseline_benchmark,
        quality_eval,
        standard_eval,
    })
}

fn benchmark_only(
    benchmark: crate::ffi::runtime_bindings::MsBaselineBenchmark,
) -> (
    crate::ffi::runtime_bindings::MsBaselineBenchmark,
    Option<crate::ffi::runtime_bindings::MsRecipeEvalResult>,
    Option<Vec<crate::ffi::runtime_bindings::MsStandardEvalTaskResult>>,
    Option<Vec<crate::ffi::runtime_bindings::MsStandardEvalSampleAudit>>,
) {
    (benchmark, None, None, None)
}

fn recipe_quality_eval_from_native(
    eval: crate::ffi::runtime_bindings::MsRecipeEvalResult,
) -> RecipeQualityEval {
    let has_baseline = eval.baseline_ppl > 0.0;
    RecipeQualityEval {
        baseline_nll: has_baseline.then_some(eval.baseline_nll),
        baseline_ppl: has_baseline.then_some(eval.baseline_ppl),
        baseline_ppl_uncertainty: has_baseline.then_some(eval.baseline_ppl_uncertainty),
        baseline_eval_ms: has_baseline.then_some(eval.baseline_eval_ms),
        baseline_vram_peak_mb: has_baseline.then_some(eval.baseline_vram_peak_mb),
        baseline_vram_allocated_mb: has_baseline.then_some(eval.baseline_vram_allocated_mb),
        recipe_nll: eval.recipe_nll,
        recipe_ppl: eval.recipe_ppl,
        recipe_ppl_uncertainty: eval.recipe_ppl_uncertainty,
        recipe_eval_ms: eval.recipe_eval_ms,
        recipe_vram_peak_mb: eval.recipe_vram_peak_mb,
        recipe_vram_allocated_mb: eval.recipe_vram_allocated_mb,
        ppl_delta: eval.ppl_delta,
        ppl_delta_percent: eval.ppl_delta_percent,
        eval_token_count: eval.eval_token_count,
        eval_sample_count: eval.eval_sample_count,
        skipped_sample_count: eval.skipped_sample_count,
    }
}

fn standard_eval_report_from_native(
    tasks: &[crate::ffi::runtime_bindings::MsStandardEvalTaskResult],
    sample_audits: &[crate::ffi::runtime_bindings::MsStandardEvalSampleAudit],
    samples: &[crate::ffi::runtime_bindings::StandardEvalSampleInput],
    quality_eval: Option<&RecipeQualityEval>,
) -> StandardEvalReport {
    let has_baseline = quality_eval
        .map(|quality| quality.baseline_ppl.is_some())
        .unwrap_or(false);

    let sample_count = tasks.iter().map(|task| task.sample_count).sum::<u64>();
    let baseline_correct = tasks
        .iter()
        .map(|task| task.baseline_correct_count)
        .sum::<u64>();
    let recipe_correct = tasks
        .iter()
        .map(|task| task.recipe_correct_count)
        .sum::<u64>();
    let correct_to_wrong_count = tasks
        .iter()
        .map(|task| task.correct_to_wrong_count)
        .sum::<u64>();
    let wrong_to_correct_count = tasks
        .iter()
        .map(|task| task.wrong_to_correct_count)
        .sum::<u64>();
    let baseline_margin_sum = tasks
        .iter()
        .map(|task| task.baseline_avg_margin * task.sample_count as f64)
        .sum::<f64>();
    let recipe_margin_sum = tasks
        .iter()
        .map(|task| task.recipe_avg_margin * task.sample_count as f64)
        .sum::<f64>();
    let baseline_accuracy = if has_baseline && sample_count > 0 {
        Some(baseline_correct as f64 / sample_count as f64)
    } else {
        None
    };
    let recipe_accuracy = if sample_count > 0 {
        recipe_correct as f64 / sample_count as f64
    } else {
        0.0
    };
    let baseline_avg_margin = if has_baseline && sample_count > 0 {
        Some(baseline_margin_sum / sample_count as f64)
    } else {
        None
    };
    let recipe_avg_margin = if sample_count > 0 {
        recipe_margin_sum / sample_count as f64
    } else {
        0.0
    };

    StandardEvalReport {
        sample_count,
        task_count: tasks.len() as u64,
        baseline_accuracy,
        recipe_accuracy,
        accuracy_delta: baseline_accuracy.map(|baseline| recipe_accuracy - baseline),
        correct_to_wrong_count,
        wrong_to_correct_count,
        baseline_avg_margin,
        recipe_avg_margin,
        margin_delta: baseline_avg_margin.map(|baseline| recipe_avg_margin - baseline),
        tasks: tasks
            .iter()
            .map(|task| StandardEvalTaskReport {
                task: crate::ffi::runtime_bindings::standard_task_name(task),
                sample_count: task.sample_count,
                baseline_correct_count: has_baseline.then_some(task.baseline_correct_count),
                recipe_correct_count: task.recipe_correct_count,
                correct_to_wrong_count: task.correct_to_wrong_count,
                wrong_to_correct_count: task.wrong_to_correct_count,
                same_prediction_count: task.same_prediction_count,
                baseline_accuracy: has_baseline.then_some(task.baseline_accuracy),
                recipe_accuracy: task.recipe_accuracy,
                accuracy_delta: has_baseline.then_some(task.accuracy_delta),
                baseline_avg_margin: has_baseline.then_some(task.baseline_avg_margin),
                recipe_avg_margin: task.recipe_avg_margin,
                margin_delta: has_baseline.then_some(task.margin_delta),
                baseline_avg_correct_nll: has_baseline.then_some(task.baseline_avg_correct_nll),
                recipe_avg_correct_nll: task.recipe_avg_correct_nll,
            })
            .collect(),
        sample_audits: standard_eval_sample_audits_from_native(sample_audits, samples),
    }
}

fn standard_eval_sample_audits_from_native(
    audits: &[crate::ffi::runtime_bindings::MsStandardEvalSampleAudit],
    samples: &[crate::ffi::runtime_bindings::StandardEvalSampleInput],
) -> Vec<StandardEvalSampleAuditReport> {
    audits
        .iter()
        .filter_map(|audit| {
            let sample = samples.get(audit.sample_index as usize)?;
            let has_baseline = audit.has_baseline != 0;
            let choice_count = (audit.choice_count as usize).min(sample.choices.len());
            let baseline_prediction_index = has_baseline.then_some(audit.baseline_prediction_index);
            let baseline_correct = has_baseline.then_some(audit.baseline_correct != 0);
            let recipe_correct = audit.recipe_correct != 0;
            let flip_type = if let Some(baseline_prediction) = baseline_prediction_index {
                if baseline_prediction != audit.recipe_prediction_index {
                    if baseline_correct == Some(true) && !recipe_correct {
                        "correct_to_wrong"
                    } else if baseline_correct == Some(false) && recipe_correct {
                        "wrong_to_correct"
                    } else {
                        "prediction_changed"
                    }
                } else {
                    "same_prediction"
                }
            } else if recipe_correct {
                "recipe_correct"
            } else {
                "recipe_wrong"
            }
            .to_string();

            let choices = (0..choice_count)
                .map(|index| {
                    let baseline_nll = has_baseline.then_some(audit.baseline_choice_nlls[index]);
                    StandardEvalChoiceAuditReport {
                        index: index as u32,
                        choice: sample.choices[index].clone(),
                        continuation: sample.continuations[index].clone(),
                        denominator: audit.choice_denominators[index],
                        baseline_nll,
                        baseline_loglikelihood: baseline_nll.map(|nll| -nll),
                        baseline_score: has_baseline.then_some(audit.baseline_choice_scores[index]),
                        recipe_nll: audit.recipe_choice_nlls[index],
                        recipe_loglikelihood: -audit.recipe_choice_nlls[index],
                        recipe_score: audit.recipe_choice_scores[index],
                    }
                })
                .collect();

            Some(StandardEvalSampleAuditReport {
                task: sample.task.clone(),
                doc_id: sample.doc_id.clone(),
                sample_index: audit.sample_index,
                prompt: sample.prompt.clone(),
                target_delimiter: sample.target_delimiter.clone(),
                gold_index: audit.gold_index,
                baseline_prediction_index,
                recipe_prediction_index: audit.recipe_prediction_index,
                baseline_correct,
                recipe_correct,
                flip_type,
                choices,
            })
        })
        .collect()
}

fn baseline_runtime_benchmark_from_native(
    eval: &crate::ffi::runtime_bindings::MsRecipeEvalResult,
    model_tensor_count: Option<u64>,
) -> Option<RuntimeBenchmark> {
    if eval.baseline_ppl <= 0.0 {
        return None;
    }

    Some(RuntimeBenchmark {
        prompt_eval_tps: eval.baseline_prompt_eval_tps,
        token_gen_tps: eval.baseline_token_gen_tps,
        ttft_ms: eval.baseline_ttft_ms,
        prompt_eval_ms: eval.baseline_prompt_eval_ms,
        generation_ms: eval.baseline_generation_ms,
        vram_peak_mb: eval.baseline_vram_peak_mb,
        vram_allocated_mb: eval.baseline_vram_allocated_mb,
        load_ms: eval.baseline_load_ms,
        elapsed_ms: eval.baseline_runtime_elapsed_ms,
        model_tensor_count,
    })
}

struct LoadedStandardSubset {
    ppl_texts: Vec<String>,
    samples: Vec<crate::ffi::runtime_bindings::StandardEvalSampleInput>,
}

fn load_standard_eval_subset(preset: StandardEvalPreset) -> Result<LoadedStandardSubset, String> {
    match preset {
        StandardEvalPreset::Quick => load_quick_standard_eval_subset(),
        StandardEvalPreset::Default => load_default_standard_eval_subset(),
    }
}

fn load_quick_standard_eval_subset() -> Result<LoadedStandardSubset, String> {
    const QUICK_STANDARD_SUBSET: &str =
        include_str!("../../../evals/lm_eval_subset.quick.generated.json");
    const DEFAULT_STANDARD_SUBSET: &str = include_str!("../../../evals/lm_eval_subset.generated.json");
    let mut subset = load_standard_eval_subset_from_json(
        QUICK_STANDARD_SUBSET,
        "generated quick lm-eval-style subset",
    )?;
    let default_subset = load_standard_eval_subset_from_json(
        DEFAULT_STANDARD_SUBSET,
        "generated lm-eval-style subset",
    )?;
    subset.ppl_texts = expand_quick_ppl_corpus(subset.ppl_texts, default_subset.ppl_texts);
    Ok(subset)
}

fn load_default_standard_eval_subset() -> Result<LoadedStandardSubset, String> {
    const STANDARD_SUBSET: &str = include_str!("../../../evals/lm_eval_subset.generated.json");
    load_standard_eval_subset_from_json(STANDARD_SUBSET, "generated lm-eval-style subset")
}

fn load_standard_eval_subset_from_json(
    contents: &str,
    label: &str,
) -> Result<LoadedStandardSubset, String> {
    let subset: StandardSubset = serde_json::from_str(contents)
        .map_err(|err| format!("failed to parse {}: {}", label, err))?;
    let ppl_texts = subset
        .ppl
        .into_iter()
        .map(|entry| entry.text)
        .filter(|text| !text.trim().is_empty())
        .collect::<Vec<_>>();
    if ppl_texts.is_empty() {
        return Err(format!("{} PPL corpus is empty", label));
    }

    let mut samples = Vec::new();
    for task in subset.tasks {
        if task.output_type != "multiple_choice" {
            return Err(format!(
                "unsupported {} output type for {}: {}",
                label, task.name, task.output_type
            ));
        }
        for sample in task.samples {
            let doc_label = sample
                .doc_id
                .as_ref()
                .map(standard_doc_id_label)
                .unwrap_or_else(|| "<unknown>".to_string());
            let target_delimiter = sample.target_delimiter.ok_or_else(|| {
                format!(
                    "{} sample missing targetDelimiter for task {} docId {}",
                    label, task.name, doc_label
                )
            })?;
            if sample.choices.len() < 2 {
                return Err(format!(
                    "{} sample for {} has fewer than two choices",
                    label, task.name
                ));
            }
            if sample.gold as usize >= sample.choices.len() {
                return Err(format!(
                    "{} sample for {} has invalid gold index",
                    label, task.name
                ));
            }
            let mut continuations = Vec::with_capacity(sample.choices.len());
            let mut choice_lengths = Vec::with_capacity(sample.choices.len());
            for choice in &sample.choices {
                if choice.is_empty() {
                    return Err(format!(
                        "{} sample for {} docId {} has an empty choice",
                        label, task.name, doc_label
                    ));
                }
                choice_lengths.push(choice.chars().count() as u64);
                continuations.push(format!("{}{}", target_delimiter, choice));
            }
            samples.push(crate::ffi::runtime_bindings::StandardEvalSampleInput {
                doc_id: doc_label,
                task: task.name.clone(),
                prompt: sample.prompt,
                target_delimiter,
                choices: sample.choices,
                continuations,
                choice_lengths,
                gold_index: sample.gold,
                normalize_by_choice_length: sample.normalize_by_choice_length,
            });
        }
    }
    if samples.is_empty() {
        return Err(format!("{} sample set is empty", label));
    }

    Ok(LoadedStandardSubset { ppl_texts, samples })
}

fn expand_quick_ppl_corpus(mut quick_ppl: Vec<String>, default_ppl: Vec<String>) -> Vec<String> {
    for text in default_ppl {
        if !quick_ppl.iter().any(|existing| existing == &text) {
            quick_ppl.push(text);
        }
    }
    quick_ppl
}

fn standard_doc_id_label(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn task_counts(subset: &LoadedStandardSubset) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for sample in &subset.samples {
            *counts.entry(sample.task.clone()).or_insert(0) += 1;
        }
        counts
    }

    #[test]
    fn default_standard_eval_uses_frozen_official_row_subset() {
        let subset = load_standard_eval_subset(StandardEvalPreset::Default).unwrap();
        let counts = task_counts(&subset);

        assert_eq!(subset.samples.len(), 600);
        assert_eq!(counts.get("arc_challenge"), Some(&100));
        assert_eq!(counts.get("arc_easy"), Some(&100));
        assert_eq!(counts.get("hellaswag"), Some(&150));
        assert_eq!(counts.get("mmlu_high_school_physics"), Some(&50));
        assert_eq!(counts.get("mmlu_college_computer_science"), Some(&50));
        assert_eq!(counts.get("mmlu_professional_medicine"), Some(&50));
        assert_eq!(counts.get("truthfulqa_mc1"), Some(&100));
        assert!(!counts.contains_key("gsm8k"));
        assert!(!counts.contains_key("mmlu_mixed"));
        assert!(!counts.contains_key("truthfulqa_mc"));
    }

    #[test]
    fn quick_standard_eval_uses_smaller_official_row_subset() {
        let subset = load_standard_eval_subset(StandardEvalPreset::Quick).unwrap();
        let counts = task_counts(&subset);

        assert_eq!(subset.samples.len(), 55);
        assert_eq!(counts.get("arc_challenge"), Some(&10));
        assert_eq!(counts.get("arc_easy"), Some(&10));
        assert_eq!(counts.get("hellaswag"), Some(&10));
        assert_eq!(counts.get("mmlu_high_school_physics"), Some(&5));
        assert_eq!(counts.get("mmlu_college_computer_science"), Some(&5));
        assert_eq!(counts.get("mmlu_professional_medicine"), Some(&5));
        assert_eq!(counts.get("truthfulqa_mc1"), Some(&10));
        assert!(!counts.contains_key("gsm8k"));
        assert!(!counts.contains_key("mmlu_mixed"));
        assert!(!counts.contains_key("truthfulqa_mc"));
    }

    #[test]
    fn quick_standard_eval_expands_only_the_ppl_corpus_from_default() {
        let quick = load_standard_eval_subset(StandardEvalPreset::Quick).unwrap();
        let default = load_standard_eval_subset(StandardEvalPreset::Default).unwrap();
        let quick_counts = task_counts(&quick);

        assert_eq!(quick.samples.len(), 55);
        assert_eq!(quick_counts.get("arc_challenge"), Some(&10));
        assert_eq!(quick_counts.get("truthfulqa_mc1"), Some(&10));
        assert!(quick.ppl_texts.len() >= default.ppl_texts.len());
        assert!(
            default
                .ppl_texts
                .iter()
                .all(|text| quick.ppl_texts.iter().any(|quick_text| quick_text == text))
        );
    }

    #[test]
    fn standard_eval_requires_explicit_target_delimiter() {
        let contents = r#"{
          "ppl": [{"text": "small fixed corpus"}],
          "tasks": [{
            "name": "arc_challenge",
            "outputType": "multiple_choice",
            "samples": [{
              "prompt": "Question: x\nAnswer:",
              "choices": [" yes", " no"],
              "gold": 0,
              "normalizeByChoiceLength": true
            }]
          }]
        }"#;

        let err = match load_standard_eval_subset_from_json(contents, "test subset") {
            Ok(_) => panic!("expected missing targetDelimiter error"),
            Err(err) => err,
        };
        assert!(
            err.contains("missing targetDelimiter"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn standard_eval_builds_delimited_continuations_and_raw_choice_lengths() {
        let contents = r#"{
          "ppl": [{"text": "small fixed corpus"}],
          "tasks": [{
            "name": "arc_challenge",
            "outputType": "multiple_choice",
            "samples": [{
              "docId": "sample-1",
              "prompt": "Question: x\nAnswer:",
              "targetDelimiter": " ",
              "choices": ["yes", "nope", "é"],
              "gold": 0,
              "normalizeByChoiceLength": true
            }]
          }]
        }"#;

        let subset = load_standard_eval_subset_from_json(contents, "test subset").unwrap();
        let sample = &subset.samples[0];
        assert_eq!(sample.continuations, vec![" yes", " nope", " é"]);
        assert_eq!(sample.choice_lengths, vec![3, 4, 1]);
    }
}
