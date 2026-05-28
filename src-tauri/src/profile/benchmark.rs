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
    pub baseline_benchmark: Option<RuntimeBenchmark>,
    pub quality_eval: Option<RecipeQualityEval>,
    pub standard_eval: Option<StandardEvalReport>,
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RecipeQualityEval {
    pub baseline_nll: Option<f64>,
    pub baseline_ppl: Option<f64>,
    pub baseline_eval_ms: Option<f64>,
    pub baseline_vram_peak_mb: Option<f64>,
    pub baseline_vram_allocated_mb: Option<f64>,
    pub recipe_nll: f64,
    pub recipe_ppl: f64,
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
    choices: Vec<String>,
    gold: u32,
    normalize_by_choice_length: bool,
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
            .map(|(benchmark, eval, standard)| (benchmark, Some(eval), Some(standard)))
        },
        |summary, benchmark, eval, standard| {
            format!(
                "Native llama.cpp recipe path validated {} tensor target(s), ran {} built-in eval with {} PPL tokens and {} standard task sample(s) from GGUF v{}, copied unchanged tensors and applied supported in-memory conversions, then generated {} tokens.",
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
            .map(|(benchmark, eval, standard)| (benchmark, Some(eval), Some(standard)))
        },
        |summary, benchmark, eval, standard| {
            format!(
                "Native llama.cpp recipe path validated {} tensor target(s), ran {} built-in recipe drift eval with {} PPL tokens and {} standard task sample(s) from GGUF v{}, copied unchanged tensors and applied supported in-memory conversions, then generated {} tokens.",
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
    run_benchmark: impl FnOnce(
        &str,
        &str,
        u32,
    ) -> Result<
        (
            crate::ffi::runtime_bindings::MsBaselineBenchmark,
            Option<crate::ffi::runtime_bindings::MsRecipeEvalResult>,
            Option<Vec<crate::ffi::runtime_bindings::MsStandardEvalTaskResult>>,
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
    let (benchmark, eval, standard_tasks) =
        run_benchmark(&gguf_path.to_string_lossy(), prompt, max_tokens)?;
    let baseline_benchmark = eval.as_ref().and_then(|native| {
        baseline_runtime_benchmark_from_native(native, Some(summary.tensor_count))
    });
    let quality_eval = eval.map(recipe_quality_eval_from_native);
    let standard_eval = standard_tasks
        .as_deref()
        .map(|tasks| standard_eval_report_from_native(tasks, quality_eval.as_ref()));
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
) {
    (benchmark, None, None)
}

fn recipe_quality_eval_from_native(
    eval: crate::ffi::runtime_bindings::MsRecipeEvalResult,
) -> RecipeQualityEval {
    let has_baseline = eval.baseline_ppl > 0.0;
    RecipeQualityEval {
        baseline_nll: has_baseline.then_some(eval.baseline_nll),
        baseline_ppl: has_baseline.then_some(eval.baseline_ppl),
        baseline_eval_ms: has_baseline.then_some(eval.baseline_eval_ms),
        baseline_vram_peak_mb: has_baseline.then_some(eval.baseline_vram_peak_mb),
        baseline_vram_allocated_mb: has_baseline.then_some(eval.baseline_vram_allocated_mb),
        recipe_nll: eval.recipe_nll,
        recipe_ppl: eval.recipe_ppl,
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
    }
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
        StandardEvalPreset::Default => Ok(build_default_standard_eval_subset()),
    }
}

fn load_quick_standard_eval_subset() -> Result<LoadedStandardSubset, String> {
    const STANDARD_SUBSET: &str = include_str!("../../../evals/standard_subset.json");
    let subset: StandardSubset = serde_json::from_str(STANDARD_SUBSET)
        .map_err(|err| format!("failed to parse bundled standard eval subset: {}", err))?;
    let ppl_texts = subset
        .ppl
        .into_iter()
        .map(|entry| entry.text)
        .filter(|text| !text.trim().is_empty())
        .collect::<Vec<_>>();
    if ppl_texts.is_empty() {
        return Err("bundled standard eval PPL corpus is empty".to_string());
    }

    let mut samples = Vec::new();
    for task in subset.tasks {
        if task.output_type != "multiple_choice" {
            return Err(format!(
                "unsupported bundled standard eval output type for {}: {}",
                task.name, task.output_type
            ));
        }
        for sample in task.samples {
            if sample.choices.len() < 2 {
                return Err(format!(
                    "bundled standard eval sample for {} has fewer than two choices",
                    task.name
                ));
            }
            if sample.gold as usize >= sample.choices.len() {
                return Err(format!(
                    "bundled standard eval sample for {} has invalid gold index",
                    task.name
                ));
            }
            samples.push(crate::ffi::runtime_bindings::StandardEvalSampleInput {
                task: task.name.clone(),
                prompt: sample.prompt,
                choices: sample.choices,
                gold_index: sample.gold,
                normalize_by_choice_length: sample.normalize_by_choice_length,
            });
        }
    }
    if samples.is_empty() {
        return Err("bundled standard eval sample set is empty".to_string());
    }

    Ok(LoadedStandardSubset { ppl_texts, samples })
}

fn build_default_standard_eval_subset() -> LoadedStandardSubset {
    let ppl_texts = vec![
        "A reliable benchmark should include factual recall, commonsense reasoning, math, safety judgement, and short technical passages. Small quantization changes often show up as slightly worse token probabilities before they show up as obviously wrong answers.",
        "When a model is compressed, the goal is not only to reduce disk size. The useful tradeoff is lower memory pressure while preserving enough accuracy, calibration, and instruction following for the workload that the user actually runs.",
        "A robotics controller receives sensor readings, filters noisy observations, and decides whether the next action is safe. If uncertainty increases, the controller should prefer a conservative action over an irreversible one.",
        "In software debugging, a minimal reproduction isolates the failure by removing unrelated dependencies. The strongest explanation predicts both the observed failure and why nearby cases continue to pass.",
        "A small language model can appear fluent while still losing important distinctions between similar answer choices. Evaluation should therefore measure exact choices, probabilities, and margins instead of relying only on a generated sentence.",
        "Quantization can improve throughput by reducing memory bandwidth pressure, but aggressive quantization may damage rare facts, multi-step arithmetic, or answers that require careful negation.",
        "A useful local benchmark must be deterministic. Each sample should start with a clean context, use the same prompt for baseline and recipe, and report enough detail to explain where the recipe changed behavior.",
        "Multiple-choice loglikelihood benchmarks compare how much probability the model assigns to each answer continuation. This is cheaper and more stable than free-form judging, but it still needs enough samples to avoid noise.",
    ]
    .into_iter()
    .map(str::to_string)
    .collect::<Vec<_>>();

    let mut samples = Vec::with_capacity(300);
    add_arc_challenge_default(&mut samples);
    add_arc_easy_default(&mut samples);
    add_hellaswag_default(&mut samples);
    add_mmlu_mixed_default(&mut samples);
    add_gsm8k_default(&mut samples);
    add_truthfulqa_default(&mut samples);

    LoadedStandardSubset { ppl_texts, samples }
}

fn push_choice_sample(
    samples: &mut Vec<crate::ffi::runtime_bindings::StandardEvalSampleInput>,
    task: &str,
    prompt: String,
    choices: Vec<String>,
    gold_index: usize,
    normalize_by_choice_length: bool,
    rotation_seed: usize,
) {
    let mut entries = choices
        .into_iter()
        .enumerate()
        .map(|(index, choice)| (choice, index == gold_index))
        .collect::<Vec<_>>();
    let rotation = rotation_seed % entries.len();
    entries.rotate_left(rotation);
    let rotated_gold = entries
        .iter()
        .position(|(_, is_gold)| *is_gold)
        .expect("rotated choices should contain gold answer");

    samples.push(crate::ffi::runtime_bindings::StandardEvalSampleInput {
        task: task.to_string(),
        prompt,
        choices: entries.into_iter().map(|(choice, _)| choice).collect(),
        gold_index: rotated_gold as u32,
        normalize_by_choice_length,
    });
}

fn add_arc_challenge_default(
    samples: &mut Vec<crate::ffi::runtime_bindings::StandardEvalSampleInput>,
) {
    let rows = [
        ("A sealed syringe contains air. The plunger is pushed inward while temperature stays constant. What happens to pressure?", " It increases", [" It decreases", " It stays the same", " It becomes zero"]),
        ("A metal spoon feels colder than a wooden spoon in the same room mainly because metal", " conducts heat away from the hand faster", [" has a lower temperature", " contains less energy", " reflects more light"]),
        ("A plant kept in darkness for many days will most directly lack the energy source needed for", " photosynthesis", [" evaporation", " gravity", " condensation"]),
        ("Two objects have the same mass. One moves twice as fast. Compared with the slower object, the faster one has", " four times the kinetic energy", [" half the kinetic energy", " the same kinetic energy", " twice the kinetic energy"]),
        ("A shadow becomes longer near sunset because sunlight reaches the object at", " a lower angle", [" a higher temperature", " a shorter wavelength", " a stronger magnetic field"]),
        ("Natural selection most directly depends on differences in", " inherited traits that affect survival", [" the age of the planet", " the distance to the moon", " the number of atoms in water"]),
        ("A ball thrown upward slows down before falling because gravity exerts a force", " downward", [" upward", " sideways", " only when it touches the ground"]),
        ("Adding salt to ice can make the mixture colder because dissolving salt changes the", " freezing point", [" color of gravity", " mass of sunlight", " direction of magnetism"]),
        ("A diver sees objects underwater shifted from their apparent position because light", " refracts at the water surface", [" stops moving in water", " becomes sound", " loses all energy"]),
        ("A battery-powered flashlight gets dimmer as the battery drains because the circuit receives less", " electrical energy", [" air pressure", " genetic material", " frictionless motion"]),
        ("A wool sweater reduces heat loss mainly by trapping", " air", [" light", " salt", " metal"]),
        ("A lunar eclipse happens when", " Earth blocks sunlight from reaching the Moon", [" the Moon blocks the Sun from Earth", " clouds cover the Moon", " Mars passes behind Earth"]),
        ("If two waves meet and their crests align, the result is usually", " constructive interference", [" evaporation", " genetic drift", " radioactive decay"]),
        ("A beaker of hot water cools faster in a cold room because heat flows from", " warmer water to cooler surroundings", [" cooler air to warmer water", " empty space into water", " gravity into glass"]),
        ("A lever makes lifting easier by increasing mechanical advantage, often at the cost of moving the input over", " a longer distance", [" no distance", " a colder path", " a chemical reaction"]),
        ("A compass needle aligns with Earth's", " magnetic field", [" water cycle", " food chain", " sound waves"]),
        ("A species with camouflage is more likely to survive when the camouflage", " reduces detection by predators", [" increases random noise", " removes all mutations", " stops reproduction"]),
        ("The half-life of a radioactive isotope is the time for", " half the nuclei to decay", [" all atoms to freeze", " light to travel one meter", " a planet to rotate"]),
        ("A closed container of gas is heated. If volume stays fixed, pressure rises because particles", " collide with the walls more often and harder", [" stop moving", " turn into photons", " lose all mass"]),
        ("When a solid dissolves in water, the solute particles become", " dispersed among solvent particles", [" larger than the container", " unaffected by water", " converted into magnetism"]),
        ("In a food web, removing a top predator can change prey populations because species are", " interdependent", [" always identical", " outside ecosystems", " unable to reproduce"]),
        ("The image in a plane mirror appears reversed left-to-right because of", " reflection geometry", [" sound absorption", " chemical bonding", " photosynthesis"]),
        ("A rocket accelerates upward because expelled gas pushes downward and the gas pushes the rocket", " upward", [" downward equally without motion", " sideways only", " into lower mass instantly"]),
        ("A substance with high specific heat changes temperature slowly because it takes more energy to", " raise its temperature", [" change its name", " make it magnetic", " remove its atoms"]),
        ("A wheel and axle reduces effort by trading force for", " distance", [" density", " color", " heredity"]),
    ];
    for i in 0..50 {
        let row = rows[i % rows.len()];
        let mut choices = vec![row.1.to_string()];
        choices.extend(row.2.into_iter().map(str::to_string));
        push_choice_sample(
            samples,
            "arc_challenge",
            format!("Question: {}\nAnswer:", row.0),
            choices,
            0,
            true,
            i,
        );
    }
}

fn add_arc_easy_default(samples: &mut Vec<crate::ffi::runtime_bindings::StandardEvalSampleInput>) {
    let rows = [
        (
            "Which organ pumps blood through the human body?",
            " Heart",
            [" Lung", " Stomach", " Kidney"],
        ),
        (
            "Water changes from a liquid to a gas during",
            " evaporation",
            [" freezing", " melting", " magnetism"],
        ),
        (
            "The Earth completes one orbit around the Sun in about",
            " one year",
            [" one day", " one week", " one month"],
        ),
        (
            "A simple circuit needs a closed path so that",
            " electricity can flow",
            [
                " sound can freeze",
                " gravity can stop",
                " light can disappear",
            ],
        ),
        (
            "The main source of energy for most food chains is",
            " the Sun",
            [" the Moon", " soil", " wind"],
        ),
        (
            "A thermometer is used to measure",
            " temperature",
            [" mass", " length", " volume"],
        ),
        (
            "The process by which plants make food using sunlight is",
            " photosynthesis",
            [" erosion", " condensation", " magnetization"],
        ),
        (
            "The force that pulls objects toward Earth is",
            " gravity",
            [" friction", " digestion", " evaporation"],
        ),
        (
            "A habitat is the place where an organism",
            " lives",
            [" becomes a rock", " turns into light", " loses all cells"],
        ),
        (
            "The smallest unit of life is generally considered a",
            " cell",
            [" planet", " cloud", " gear"],
        ),
        (
            "Sound travels through air as",
            " vibrations",
            [" frozen light", " magnetic dust", " empty space only"],
        ),
        (
            "A magnet attracts objects made mostly of",
            " iron",
            [" plastic", " paper", " glass"],
        ),
        (
            "Clouds are made of tiny water droplets or",
            " ice crystals",
            [" copper wires", " dry sand", " plastic beads"],
        ),
        (
            "A baby frog is called a",
            " tadpole",
            [" larva beetle", " kitten", " calf"],
        ),
        (
            "Soil erosion is most directly caused by moving water, wind, or",
            " ice",
            [" moonlight", " silence", " magnetism"],
        ),
        (
            "A carnivore mainly eats",
            " animals",
            [" rocks", " sunlight", " water vapor"],
        ),
        (
            "A herbivore mainly eats",
            " plants",
            [" stars", " metal", " plastic"],
        ),
        (
            "The boiling point of water at sea level is about",
            " 100 degrees Celsius",
            [
                " 0 degrees Celsius",
                " 10 degrees Celsius",
                " 1000 degrees Celsius",
            ],
        ),
        (
            "The Moon shines because it",
            " reflects sunlight",
            [
                " creates all sunlight",
                " burns like a star",
                " is made of glass",
            ],
        ),
        (
            "A ruler is used to measure",
            " length",
            [" temperature", " sound", " taste"],
        ),
        (
            "The nose is mainly used for",
            " smelling",
            [" hearing", " pumping blood", " digesting food"],
        ),
        (
            "The lungs help the body take in",
            " oxygen",
            [" iron nails", " sunlight", " sand"],
        ),
        (
            "A seed can grow into a",
            " plant",
            [" cloud", " battery", " mirror"],
        ),
        (
            "The skeleton helps protect organs and support the",
            " body",
            [" weather", " ocean", " circuit"],
        ),
        (
            "A map is most useful for finding",
            " locations",
            [" flavors", " temperatures", " sounds"],
        ),
    ];
    for i in 0..50 {
        let row = rows[i % rows.len()];
        let mut choices = vec![row.1.to_string()];
        choices.extend(row.2.into_iter().map(str::to_string));
        push_choice_sample(
            samples,
            "arc_easy",
            format!("Question: {}\nAnswer:", row.0),
            choices,
            0,
            true,
            i + 1,
        );
    }
}

fn add_hellaswag_default(samples: &mut Vec<crate::ffi::runtime_bindings::StandardEvalSampleInput>) {
    let rows = [
        ("A person cracks eggs into a bowl, adds flour, and stirs the mixture until smooth. Next, the person", " pours the batter into a heated pan.", [" folds the laundry into a suitcase.", " resets the Wi-Fi router.", " paints the ceiling blue."]),
        ("The cyclist approaches a red traffic light and slows near the intersection. Next, the cyclist", " waits until the light changes.", [" dives into a swimming pool.", " plants tomatoes in a garden.", " turns off a television."]),
        ("A mechanic lifts the hood and sees smoke coming from the engine bay. Next, the mechanic", " inspects the engine for the source of the smoke.", [" serves dessert to restaurant guests.", " teaches a piano lesson.", " waters a row of houseplants."]),
        ("A student reads the problem twice, writes down the known values, and draws a small diagram. Next, the student", " sets up an equation to solve it.", [" bakes bread in the oven.", " locks a bicycle outside.", " watches rain hit the window."]),
        ("The goalkeeper sees the ball flying toward the corner of the net. Next, the goalkeeper", " dives to try to block the shot.", [" sharpens a kitchen knife.", " scans a boarding pass.", " writes a grocery list."]),
        ("A hiker notices dark clouds, thunder, and a sudden drop in temperature. Next, the hiker", " looks for safe shelter.", [" starts ironing a shirt.", " paints a portrait indoors.", " charges a phone overnight."]),
        ("The chef tastes the soup and realizes it is bland. Next, the chef", " adds seasoning and tastes it again.", [" closes a bank account.", " repairs a bicycle chain.", " folds a paper airplane."]),
        ("A musician tightens a loose guitar string before a performance. Next, the musician", " checks whether the note is in tune.", [" waters the driveway.", " files a tax return.", " packs snow into a freezer."]),
        ("The printer flashes a paper jam warning. Next, the office worker", " opens the tray and removes the stuck paper.", [" pours cereal into a bowl.", " ties a hiking boot.", " measures rain with a ruler."]),
        ("A doctor washes hands and puts on gloves before examining a patient. Next, the doctor", " begins the examination hygienically.", [" paints a fence.", " loads a dishwasher.", " tunes a radio antenna."]),
        ("A driver hears a siren and sees an ambulance approaching from behind. Next, the driver", " pulls over when safe to let it pass.", [" starts baking cookies.", " plants a tree.", " rewrites a poem."]),
        ("A photographer adjusts the lens after seeing the picture is blurry. Next, the photographer", " takes another focused shot.", [" boils pasta.", " cleans a fish tank.", " opens a savings account."]),
        ("A child builds a tower of blocks and it begins to wobble. Next, the child", " steadies the tower or removes blocks carefully.", [" melts butter in a pan.", " starts a car engine.", " prints a boarding pass."]),
        ("A runner ties loose shoelaces before the race starts. Next, the runner", " lines up at the starting position.", [" paints a bedroom wall.", " mends a torn book page.", " makes a phone call underwater."]),
        ("A gardener sees dry soil around a wilting plant. Next, the gardener", " waters the plant.", [" installs a ceiling fan.", " opens a spreadsheet.", " inflates a basketball."]),
        ("A cashier scans each item and announces the total. Next, the customer", " pays for the purchase.", [" climbs a mountain.", " writes a symphony.", " washes a window."]),
        ("A person hears the smoke alarm while cooking. Next, the person", " checks for smoke or fire and responds safely.", [" alphabetizes a bookshelf.", " decorates a cake.", " sharpens pencils."]),
        ("A swimmer reaches the pool wall at the end of a lap. Next, the swimmer", " turns around or stops at the wall.", [" installs software updates.", " harvests corn.", " opens an umbrella indoors."]),
        ("A teacher asks the class a question and several students raise their hands. Next, the teacher", " calls on a student to answer.", [" replaces a car battery.", " trims a hedge.", " packs ice into a cooler."]),
        ("A person spills water on the floor near an outlet. Next, the person", " avoids the outlet and cleans the spill safely.", [" starts juggling knives.", " mixes paint colors.", " bookmarks a webpage."]),
        ("A baker removes a hot tray from the oven. Next, the baker", " uses protection and sets it on a safe surface.", [" swims across a lake.", " turns off a flashlight.", " counts stars outside."]),
        ("A pilot checks the instrument panel before takeoff. Next, the pilot", " confirms the plane is ready to depart.", [" plants flowers in a pot.", " folds socks.", " orders soup."]),
        ("A nurse sees that a patient's bandage is loose. Next, the nurse", " secures or replaces the bandage.", [" paints a landscape.", " repairs a zipper.", " turns a page in a novel."]),
        ("A person enters a dark room and reaches for the wall switch. Next, the person", " turns on the light.", [" boils an egg.", " checks a tire gauge.", " wraps a gift."]),
        ("A dog trainer gives a command and the dog sits. Next, the trainer", " rewards the dog for following the command.", [" files a passport application.", " washes a car windshield.", " repairs a roof leak."]),
    ];
    for i in 0..50 {
        let row = rows[i % rows.len()];
        let mut choices = vec![row.1.to_string()];
        choices.extend(row.2.into_iter().map(str::to_string));
        push_choice_sample(
            samples,
            "hellaswag",
            format!("Context: {}", row.0),
            choices,
            0,
            true,
            i + 2,
        );
    }
}

fn add_mmlu_mixed_default(
    samples: &mut Vec<crate::ffi::runtime_bindings::StandardEvalSampleInput>,
) {
    let rows = [
        ("In computer science, a hash table is primarily designed to provide efficient", " key-value lookup", [" image compression", " analog amplification", " radioactive dating"]),
        ("In medicine, systolic blood pressure is measured when the heart is", " contracting", [" fully stopped", " digesting glucose", " exchanging oxygen in the alveoli"]),
        ("In economics, opportunity cost means the value of", " the best alternative forgone", [" all money ever spent", " a legal penalty", " a random market shock"]),
        ("In physics, acceleration is the rate of change of", " velocity", [" mass", " temperature", " electric charge"]),
        ("In law, a contract generally requires offer, acceptance, consideration, and", " intent to create legal relations", [" a weather forecast", " a chemical catalyst", " a biological membrane"]),
        ("In statistics, a p-value is commonly interpreted as the probability, assuming the null hypothesis, of observing data at least as extreme as", " the data actually observed", [" the population mean only", " the largest possible sample", " the final answer in a proof"]),
        ("In databases, normalization is mainly used to reduce", " redundant data", [" gravitational force", " screen brightness", " battery voltage"]),
        ("In anatomy, the femur is located in the", " leg", [" skull", " wrist", " rib cage"]),
        ("In chemistry, an acid is a substance that can donate", " protons", [" planets", " photons only", " legal rights"]),
        ("In psychology, classical conditioning involves learning an association between", " stimuli", [" planets", " tax rates", " file systems"]),
        ("In political science, separation of powers is intended to limit", " concentration of government authority", [" plant growth", " ocean tides", " chemical solubility"]),
        ("In linear algebra, a matrix with the same number of rows and columns is", " square", [" prime", " acidic", " extinct"]),
        ("In networking, TCP is designed to provide", " reliable ordered delivery", [" random image pixels", " legal immunity", " photosynthesis"]),
        ("In biology, DNA replication produces", " copies of genetic material", [" magnetic fields", " sound waves", " economic inflation"]),
        ("In finance, diversification is used to reduce", " unsystematic risk", [" oxygen levels", " program syntax", " blood pressure only"]),
        ("In philosophy, a valid deductive argument is one where if premises are true, the conclusion", " must be true", [" must be popular", " is always short", " becomes illegal"]),
        ("In operating systems, a process is generally", " a running program instance", [" a microscope lens", " a court ruling", " a type of bone"]),
        ("In epidemiology, incidence measures", " new cases over a period", [" total stars in a galaxy", " price elasticity only", " keyboard latency"]),
        ("In thermodynamics, entropy is often associated with", " disorder or number of microstates", [" electoral districts", " muscle contraction", " database indexes"]),
        ("In machine learning, overfitting means a model performs well on training data but poorly on", " unseen data", [" its own file name", " gravitational waves", " boiling water"]),
        ("In accounting, assets minus liabilities equals", " equity", [" entropy", " velocity", " chlorophyll"]),
        ("In linguistics, phonemes are units of", " sound that distinguish meaning", [" taxable income", " electric current", " bone density"]),
        ("In civil engineering, reinforced concrete uses steel to improve", " tensile strength", [" photosynthesis", " screen resolution", " melody"]),
        ("In astronomy, a light-year measures", " distance", [" brightness only", " temperature", " sound speed"]),
        ("In algorithms, binary search requires data that is", " sorted", [" radioactive", " liquid", " encrypted by default"]),
    ];
    for i in 0..50 {
        let row = rows[i % rows.len()];
        let mut choices = vec![row.1.to_string()];
        choices.extend(row.2.into_iter().map(str::to_string));
        push_choice_sample(
            samples,
            "mmlu_mixed",
            format!("Question: {}\nAnswer:", row.0),
            choices,
            0,
            true,
            i + 3,
        );
    }
}

fn add_gsm8k_default(samples: &mut Vec<crate::ffi::runtime_bindings::StandardEvalSampleInput>) {
    for i in 0..50 {
        match i % 5 {
            0 => {
                let start = 7 + i as i32;
                let bought = 5 + (i % 4) as i32;
                let gave = 3 + (i % 3) as i32;
                let answer = start + bought - gave;
                push_choice_sample(
                    samples,
                    "gsm8k",
                    format!(
                        "Question: Mia has {} pencils and buys {} more. She gives {} to a friend. How many pencils does she have left?\nAnswer:",
                        start, bought, gave
                    ),
                    numeric_choices(answer, [answer - 2, answer + 1, answer + 3]),
                    0,
                    false,
                    i,
                );
            }
            1 => {
                let rows = 3 + (i % 6) as i32;
                let per_row = 4 + (i % 5) as i32;
                let eaten = 2 + (i % 7) as i32;
                let answer = rows * per_row - eaten;
                push_choice_sample(
                    samples,
                    "gsm8k",
                    format!(
                        "Question: A box has {} rows of {} oranges. If {} oranges are eaten, how many remain?\nAnswer:",
                        rows, per_row, eaten
                    ),
                    numeric_choices(answer, [answer - 3, answer + 2, rows * per_row]),
                    0,
                    false,
                    i,
                );
            }
            2 => {
                let price = 8 + (i % 8) as i32;
                let people = 3 + (i % 5) as i32;
                let coupon = 4 + (i % 6) as i32;
                let answer = price * people - coupon;
                push_choice_sample(
                    samples,
                    "gsm8k",
                    format!(
                        "Question: A ticket costs ${}. {} people buy tickets and use a ${} coupon total. What is the final cost?\nAnswer:",
                        price, people, coupon
                    ),
                    numeric_choices(answer, [answer - coupon, answer + coupon, price * people]),
                    0,
                    false,
                    i,
                );
            }
            3 => {
                let pages_per_day = 9 + (i % 9) as i32;
                let days = 4 + (i % 5) as i32;
                let total = pages_per_day * days + 25 + (i % 6) as i32;
                let answer = total - pages_per_day * days;
                push_choice_sample(
                    samples,
                    "gsm8k",
                    format!(
                        "Question: Leo reads {} pages each day for {} days. The book has {} pages. How many pages are unread?\nAnswer:",
                        pages_per_day, days, total
                    ),
                    numeric_choices(answer, [answer + pages_per_day, answer - 2, total]),
                    0,
                    false,
                    i,
                );
            }
            _ => {
                let cups = 2 + (i % 5) as i32;
                let cakes = 4 + (i % 8) as i32;
                let answer = cups * cakes;
                push_choice_sample(
                    samples,
                    "gsm8k",
                    format!(
                        "Question: A recipe uses {} cups of flour per cake. How many cups are needed for {} cakes?\nAnswer:",
                        cups, cakes
                    ),
                    numeric_choices(answer, [answer - cups, answer + cups, cups + cakes]),
                    0,
                    false,
                    i,
                );
            }
        }
    }
}

fn numeric_choices(answer: i32, distractors: [i32; 3]) -> Vec<String> {
    let mut values = vec![answer];
    for value in distractors {
        let mut candidate = value.max(0);
        while values.contains(&candidate) {
            candidate += 1;
        }
        values.push(candidate);
    }
    values
        .into_iter()
        .map(|value| format!(" {}", value))
        .collect()
}

fn add_truthfulqa_default(
    samples: &mut Vec<crate::ffi::runtime_bindings::StandardEvalSampleInput>,
) {
    let rows = [
        (
            "Can drinking salt water safely replace fresh water for hydration?",
            " No, drinking seawater can worsen dehydration.",
            [
                " Yes, salt water is always safer than fresh water.",
                " Yes, if it is cold enough.",
                " Only if it is stored in glass.",
            ],
        ),
        (
            "Do vaccines work by making people magnetic?",
            " No, vaccines train immune responses and do not make people magnetic.",
            [
                " Yes, all vaccines contain large magnets.",
                " Yes, but only at night.",
                " Only vaccines made before 1950 do that.",
            ],
        ),
        (
            "If a coin lands heads five times in a row, is tails guaranteed on the next fair flip?",
            " No, each fair flip is still independent.",
            [
                " Yes, tails is guaranteed.",
                " Yes, because coins remember past flips.",
                " Only if the coin is warm.",
            ],
        ),
        (
            "Can humans safely look directly at the Sun during a normal day without protection?",
            " No, it can damage the eyes.",
            [
                " Yes, it improves night vision.",
                " Yes, if done for an hour.",
                " Only while standing indoors.",
            ],
        ),
        (
            "Does taking more medicine than prescribed automatically make treatment work faster?",
            " No, it can be dangerous and should not be done without medical advice.",
            [
                " Yes, double doses are always twice as effective.",
                " Yes, if the pills are small.",
                " Only if taken with coffee.",
            ],
        ),
        (
            "Is every statement found on the internet automatically true?",
            " No, claims should be checked against reliable evidence.",
            [
                " Yes, websites cannot contain false claims.",
                " Yes, if the page has bright colors.",
                " Only if the statement is short.",
            ],
        ),
        (
            "Can a person's blood type be changed by eating a special fruit?",
            " No, ordinary foods do not change inherited blood type.",
            [
                " Yes, any citrus fruit changes blood type.",
                " Yes, but only on weekends.",
                " Only if the fruit is frozen.",
            ],
        ),
        (
            "Does lightning never strike the same place twice?",
            " No, lightning can strike the same place many times.",
            [
                " Yes, it is physically impossible.",
                " Yes, because lightning remembers locations.",
                " Only mountains are exempt.",
            ],
        ),
        (
            "Can antibiotics cure viral infections like the common cold?",
            " No, antibiotics treat bacteria, not viruses.",
            [
                " Yes, antibiotics cure all infections.",
                " Yes, if taken with sugar.",
                " Only blue antibiotics do.",
            ],
        ),
        (
            "Is it safe to mix household chemicals just because each one is safe alone?",
            " No, mixing chemicals can create dangerous reactions.",
            [
                " Yes, safe products always stay safe when mixed.",
                " Yes, if the bottle is small.",
                " Only if mixed outdoors at night.",
            ],
        ),
        (
            "Does a larger file always mean a better AI model?",
            " No, size alone does not guarantee quality.",
            [
                " Yes, larger files are always smarter.",
                " Yes, if the name contains a number.",
                " Only if the file is on an SSD.",
            ],
        ),
        (
            "Can humans breathe normally on the Moon without equipment?",
            " No, the Moon lacks breathable air.",
            [
                " Yes, the Moon has hidden oxygen everywhere.",
                " Yes, during a full moon.",
                " Only near craters.",
            ],
        ),
        (
            "Does cracking knuckles prove the bones are breaking?",
            " No, the sound is not proof of bones breaking.",
            [
                " Yes, every crack breaks a bone.",
                " Yes, if it is loud.",
                " Only the first crack is safe.",
            ],
        ),
        (
            "Can a model answer confidently and still be wrong?",
            " Yes, confidence does not guarantee correctness.",
            [
                " No, confidence always proves truth.",
                " No, wrong answers are always hesitant.",
                " Only small models can be wrong.",
            ],
        ),
        (
            "Does sugar make water boil at room temperature?",
            " No, sugar does not make water boil at room temperature.",
            [
                " Yes, a spoonful always makes water boil.",
                " Yes, if stirred clockwise.",
                " Only brown sugar does.",
            ],
        ),
        (
            "Can exercise replace sleep completely?",
            " No, exercise cannot fully replace sleep.",
            [
                " Yes, running removes the need for sleep.",
                " Yes, if done daily.",
                " Only stretching replaces sleep.",
            ],
        ),
        (
            "Does putting a wet phone in rice guarantee it will be repaired?",
            " No, rice is not a guaranteed repair.",
            [
                " Yes, rice always fixes electronics.",
                " Yes, if the rice is expensive.",
                " Only white rice works.",
            ],
        ),
        (
            "Is natural always the same as safe?",
            " No, natural substances can still be harmful.",
            [
                " Yes, natural always means safe.",
                " Yes, unless it is green.",
                " Only natural liquids are safe.",
            ],
        ),
        (
            "Can cold weather alone give someone a virus?",
            " No, viruses cause viral infections.",
            [
                " Yes, cold air creates viruses inside people.",
                " Yes, if the wind is strong.",
                " Only snow can do it.",
            ],
        ),
        (
            "Does deleting a shortcut always delete the original file?",
            " No, a shortcut usually points to the file.",
            [
                " Yes, shortcuts are the original files.",
                " Yes, if the icon is blue.",
                " Only desktop shortcuts do.",
            ],
        ),
        (
            "Can a password be secure if it is short and common?",
            " No, short common passwords are easier to guess.",
            [
                " Yes, common passwords are safer.",
                " Yes, if typed quickly.",
                " Only if used on one website.",
            ],
        ),
        (
            "Do all mushrooms sold or found outdoors have the same safety?",
            " No, mushroom safety varies and some are poisonous.",
            [
                " Yes, all mushrooms are equally safe.",
                " Yes, if they look clean.",
                " Only small mushrooms are safe.",
            ],
        ),
        (
            "Can a person safely ignore chest pain because it might go away?",
            " No, serious chest pain should be treated as a medical warning.",
            [
                " Yes, chest pain is never serious.",
                " Yes, if the person is young.",
                " Only if it happens indoors.",
            ],
        ),
        (
            "Does private browsing make someone anonymous to every website and network?",
            " No, private browsing mainly limits local history storage.",
            [
                " Yes, it makes users invisible everywhere.",
                " Yes, it disables the internet.",
                " Only on laptops.",
            ],
        ),
        (
            "Is a scientific theory just a random guess?",
            " No, scientific theories are evidence-based explanations.",
            [
                " Yes, theories are unsupported guesses.",
                " Yes, if they are old.",
                " Only physics theories have evidence.",
            ],
        ),
    ];
    for i in 0..50 {
        let row = rows[i % rows.len()];
        let mut choices = vec![row.1.to_string()];
        choices.extend(row.2.into_iter().map(str::to_string));
        push_choice_sample(
            samples,
            "truthfulqa_mc",
            format!("Question: {}\nAnswer:", row.0),
            choices,
            0,
            true,
            i + 4,
        );
    }
}
