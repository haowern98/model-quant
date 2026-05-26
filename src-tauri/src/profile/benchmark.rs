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
    pub task_eval: Option<TaskEvalSummary>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalSuite {
    OfficialCore,
    PplSmoke,
    StandardSubset,
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskEvalSummary {
    pub suite: String,
    pub tasks: Vec<TaskEvalResult>,
    pub aggregate: TaskEvalAggregate,
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskEvalResult {
    pub task: String,
    pub metric: String,
    pub sample_count: u64,
    pub baseline_score: Option<f64>,
    pub recipe_score: f64,
    pub delta: Option<f64>,
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskEvalAggregate {
    pub sample_count: u64,
    pub baseline_score: Option<f64>,
    pub recipe_score: f64,
    pub delta: Option<f64>,
}

#[derive(Debug, serde::Deserialize)]
struct EvalText {
    text: String,
}

#[derive(Debug, serde::Deserialize)]
struct StandardEvalSample {
    task: String,
    #[serde(rename = "type")]
    sample_type: String,
    prompt: String,
    choices: Option<Vec<String>>,
    answer_index: Option<u32>,
    answers: Option<Vec<String>>,
    max_tokens: Option<u32>,
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
        task_eval: None,
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
        task_eval: None,
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
    run_native_recipe_compare_benchmark(gguf_path, targets, max_tokens, progress)
}

pub fn run_native_recipe_single_benchmark(
    gguf_path: &PathBuf,
    targets: &[(String, String)],
    max_tokens: u32,
    progress: &ProgressEmitter,
) -> Result<BenchmarkResult, String> {
    run_native_recipe_single_eval(
        gguf_path,
        targets,
        max_tokens,
        progress,
        EvalSuite::PplSmoke,
    )
}

pub fn run_native_recipe_single_eval(
    gguf_path: &PathBuf,
    targets: &[(String, String)],
    max_tokens: u32,
    progress: &ProgressEmitter,
    eval_suite: EvalSuite,
) -> Result<BenchmarkResult, String> {
    if eval_suite == EvalSuite::StandardSubset {
        return run_native_recipe_standard_single_eval(gguf_path, targets, max_tokens, progress);
    }

    let eval_texts = load_smoke_eval_texts()?;
    run_native_inference_benchmark(
        gguf_path,
        max_tokens,
        progress,
        "Running single recipe model test...",
        "Native single recipe test complete",
        "native_recipe_single_v1",
        |path, prompt, max_tokens| {
            crate::ffi::runtime_bindings::eval_recipe_single(
                path,
                targets,
                &eval_texts,
                128,
                prompt,
                max_tokens,
            )
            .map(|(benchmark, eval)| (benchmark, Some(eval), None))
        },
        |summary, benchmark, eval, _| {
            format!(
                "Native llama.cpp recipe path validated {} tensor target(s), evaluated standalone quality on {} tokens from GGUF v{}, copied unchanged tensors and applied supported in-memory conversions, then generated {} tokens.",
                targets.len(),
                eval.map(|quality| quality.eval_token_count).unwrap_or(0),
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
    progress: &ProgressEmitter,
) -> Result<BenchmarkResult, String> {
    run_native_recipe_compare_eval(
        gguf_path,
        targets,
        max_tokens,
        progress,
        EvalSuite::PplSmoke,
    )
}

pub fn run_native_recipe_compare_eval(
    gguf_path: &PathBuf,
    targets: &[(String, String)],
    max_tokens: u32,
    progress: &ProgressEmitter,
    eval_suite: EvalSuite,
) -> Result<BenchmarkResult, String> {
    if eval_suite == EvalSuite::StandardSubset {
        return run_native_recipe_standard_compare_eval(gguf_path, targets, max_tokens, progress);
    }

    let eval_texts = load_smoke_eval_texts()?;
    run_native_inference_benchmark(
        gguf_path,
        max_tokens,
        progress,
        "Running baseline and recipe drift eval...",
        "Native recipe eval complete",
        "native_recipe_eval_v1",
        |path, prompt, max_tokens| {
            crate::ffi::runtime_bindings::eval_recipe(
                path,
                targets,
                &eval_texts,
                128,
                prompt,
                max_tokens,
            )
            .map(|(benchmark, eval)| (benchmark, Some(eval), None))
        },
        |summary, benchmark, eval, _| {
            format!(
                "Native llama.cpp recipe path validated {} tensor target(s), evaluated recipe drift on {} tokens from GGUF v{}, copied unchanged tensors and applied supported in-memory conversions, then generated {} tokens.",
                targets.len(),
                eval.map(|quality| quality.eval_token_count).unwrap_or(0),
                summary.version,
                benchmark.generated_tokens
            )
        },
    )
}

fn run_native_recipe_standard_single_eval(
    gguf_path: &PathBuf,
    targets: &[(String, String)],
    max_tokens: u32,
    progress: &ProgressEmitter,
) -> Result<BenchmarkResult, String> {
    let eval_texts = load_smoke_eval_texts()?;
    let samples = load_standard_eval_samples()?;
    let sample_meta = samples
        .iter()
        .map(|sample| (sample.task_name.clone(), task_metric(sample)))
        .collect::<Vec<_>>();

    run_native_inference_benchmark(
        gguf_path,
        max_tokens,
        progress,
        "Running standard task eval on recipe model...",
        "Native standard task eval complete",
        "native_standard_eval_single_v1",
        |path, prompt, max_tokens| {
            crate::ffi::runtime_bindings::eval_task_suite_single(
                path,
                targets,
                &eval_texts,
                &samples,
                128,
                prompt,
                max_tokens,
            )
            .map(|(benchmark, eval, recipe_results)| {
                let task_eval =
                    aggregate_task_results("standard_subset", &sample_meta, None, &recipe_results);
                (benchmark, Some(eval), Some(task_eval))
            })
        },
        |summary, benchmark, eval, task_eval| {
            format!(
                "Native llama.cpp standard eval ran {} task sample(s) from GGUF v{}, evaluated {} PPL tokens, copied unchanged tensors and applied supported in-memory conversions, then generated {} tokens.",
                task_eval.map(|tasks| tasks.aggregate.sample_count).unwrap_or(0),
                summary.version,
                eval.map(|quality| quality.eval_token_count).unwrap_or(0),
                benchmark.generated_tokens
            )
        },
    )
}

fn run_native_recipe_standard_compare_eval(
    gguf_path: &PathBuf,
    targets: &[(String, String)],
    max_tokens: u32,
    progress: &ProgressEmitter,
) -> Result<BenchmarkResult, String> {
    let eval_texts = load_smoke_eval_texts()?;
    let samples = load_standard_eval_samples()?;
    let sample_meta = samples
        .iter()
        .map(|sample| (sample.task_name.clone(), task_metric(sample)))
        .collect::<Vec<_>>();

    run_native_inference_benchmark(
        gguf_path,
        max_tokens,
        progress,
        "Running baseline and recipe standard task eval...",
        "Native standard comparison eval complete",
        "native_standard_eval_compare_v1",
        |path, prompt, max_tokens| {
            crate::ffi::runtime_bindings::eval_task_suite_compare(
                path,
                targets,
                &eval_texts,
                &samples,
                128,
                prompt,
                max_tokens,
            )
            .map(|(benchmark, eval, baseline_results, recipe_results)| {
                let task_eval = aggregate_task_results(
                    "standard_subset",
                    &sample_meta,
                    Some(&baseline_results),
                    &recipe_results,
                );
                (benchmark, Some(eval), Some(task_eval))
            })
        },
        |summary, benchmark, eval, task_eval| {
            format!(
                "Native llama.cpp standard eval compared baseline and recipe on {} task sample(s) from GGUF v{}, evaluated {} PPL tokens, copied unchanged tensors and applied supported in-memory conversions, then generated {} tokens.",
                task_eval.map(|tasks| tasks.aggregate.sample_count).unwrap_or(0),
                summary.version,
                eval.map(|quality| quality.eval_token_count).unwrap_or(0),
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
            Option<TaskEvalSummary>,
        ),
        String,
    >,
    status_message: impl FnOnce(
        &crate::ffi::runtime_bindings::MsGgufSummary,
        &crate::ffi::runtime_bindings::MsBaselineBenchmark,
        Option<&RecipeQualityEval>,
        Option<&TaskEvalSummary>,
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
    let (benchmark, eval, task_eval) =
        run_benchmark(&gguf_path.to_string_lossy(), prompt, max_tokens)?;
    let baseline_benchmark = eval.as_ref().and_then(|native| {
        baseline_runtime_benchmark_from_native(native, Some(summary.tensor_count))
    });
    let quality_eval = eval.map(recipe_quality_eval_from_native);
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
            task_eval.as_ref(),
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
        task_eval,
    })
}

fn benchmark_only(
    benchmark: crate::ffi::runtime_bindings::MsBaselineBenchmark,
) -> (
    crate::ffi::runtime_bindings::MsBaselineBenchmark,
    Option<crate::ffi::runtime_bindings::MsRecipeEvalResult>,
    Option<TaskEvalSummary>,
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

fn load_standard_eval_samples() -> Result<Vec<crate::ffi::runtime_bindings::EvalSample>, String> {
    const STANDARD_SUBSET: &str = include_str!("../../../evals/standard_subset.json");
    let samples: Vec<StandardEvalSample> = serde_json::from_str(STANDARD_SUBSET)
        .map_err(|err| format!("failed to parse bundled standard eval: {}", err))?;
    samples
        .into_iter()
        .map(|sample| match sample.sample_type.as_str() {
            "multiple_choice" => {
                let choices = sample
                    .choices
                    .ok_or_else(|| format!("{} sample is missing choices", sample.task))?;
                let answer_index = sample
                    .answer_index
                    .ok_or_else(|| format!("{} sample is missing answer index", sample.task))?;
                Ok(crate::ffi::runtime_bindings::EvalSample {
                    task_name: sample.task,
                    prompt: sample.prompt,
                    kind: crate::ffi::runtime_bindings::EvalSampleKind::MultipleChoice {
                        choices,
                        answer_index,
                    },
                })
            }
            "exact_match" => {
                let answers = sample
                    .answers
                    .ok_or_else(|| format!("{} sample is missing answers", sample.task))?;
                Ok(crate::ffi::runtime_bindings::EvalSample {
                    task_name: sample.task,
                    prompt: sample.prompt,
                    kind: crate::ffi::runtime_bindings::EvalSampleKind::ExactMatch {
                        answers,
                        max_tokens: sample.max_tokens.unwrap_or(16),
                    },
                })
            }
            other => Err(format!("unknown standard eval sample type: {}", other)),
        })
        .collect()
}

fn task_metric(sample: &crate::ffi::runtime_bindings::EvalSample) -> &'static str {
    match &sample.kind {
        crate::ffi::runtime_bindings::EvalSampleKind::MultipleChoice { .. } => "accuracy",
        crate::ffi::runtime_bindings::EvalSampleKind::ExactMatch { .. } => "exact_match",
    }
}

fn aggregate_task_results(
    suite: &str,
    sample_meta: &[(String, &'static str)],
    baseline_results: Option<&[crate::ffi::runtime_bindings::MsEvalSampleResult]>,
    recipe_results: &[crate::ffi::runtime_bindings::MsEvalSampleResult],
) -> TaskEvalSummary {
    #[derive(Default)]
    struct Bucket {
        metric: &'static str,
        sample_count: u64,
        baseline_correct: u64,
        recipe_correct: u64,
    }

    let mut buckets: std::collections::BTreeMap<String, Bucket> = std::collections::BTreeMap::new();
    for (index, (task, metric)) in sample_meta.iter().enumerate() {
        let bucket = buckets.entry(task.clone()).or_insert_with(|| Bucket {
            metric,
            ..Default::default()
        });
        bucket.sample_count += 1;
        if let Some(results) = baseline_results {
            if results
                .get(index)
                .map_or(false, |result| result.correct != 0)
            {
                bucket.baseline_correct += 1;
            }
        }
        if recipe_results
            .get(index)
            .map_or(false, |result| result.correct != 0)
        {
            bucket.recipe_correct += 1;
        }
    }

    let has_baseline = baseline_results.is_some();
    let mut aggregate_samples = 0_u64;
    let mut aggregate_baseline_correct = 0_u64;
    let mut aggregate_recipe_correct = 0_u64;
    let tasks = buckets
        .into_iter()
        .map(|(task, bucket)| {
            aggregate_samples += bucket.sample_count;
            aggregate_baseline_correct += bucket.baseline_correct;
            aggregate_recipe_correct += bucket.recipe_correct;
            let baseline_score = has_baseline
                .then(|| bucket.baseline_correct as f64 / bucket.sample_count.max(1) as f64);
            let recipe_score = bucket.recipe_correct as f64 / bucket.sample_count.max(1) as f64;
            TaskEvalResult {
                task,
                metric: bucket.metric.to_string(),
                sample_count: bucket.sample_count,
                baseline_score,
                recipe_score,
                delta: baseline_score.map(|score| recipe_score - score),
            }
        })
        .collect::<Vec<_>>();

    let baseline_score =
        has_baseline.then(|| aggregate_baseline_correct as f64 / aggregate_samples.max(1) as f64);
    let recipe_score = aggregate_recipe_correct as f64 / aggregate_samples.max(1) as f64;

    TaskEvalSummary {
        suite: suite.to_string(),
        tasks,
        aggregate: TaskEvalAggregate {
            sample_count: aggregate_samples,
            baseline_score,
            recipe_score,
            delta: baseline_score.map(|score| recipe_score - score),
        },
    }
}

fn load_smoke_eval_texts() -> Result<Vec<String>, String> {
    const SMOKE_TEXTS: &str = include_str!("../../../evals/smoke_texts.json");
    let texts: Vec<EvalText> = serde_json::from_str(SMOKE_TEXTS)
        .map_err(|err| format!("failed to parse bundled eval texts: {}", err))?;
    let texts = texts
        .into_iter()
        .map(|entry| entry.text)
        .filter(|text| !text.trim().is_empty())
        .collect::<Vec<_>>();
    if texts.is_empty() {
        return Err("bundled eval text set is empty".to_string());
    }
    Ok(texts)
}
