#![allow(dead_code)]

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MsGgufSummary {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_count: u64,
    pub alignment: u64,
    pub data_offset: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MsBaselineBenchmark {
    pub load_ms: f64,
    pub prompt_eval_ms: f64,
    pub generation_ms: f64,
    pub prompt_eval_tps: f64,
    pub token_gen_tps: f64,
    pub ttft_ms: f64,
    pub vram_peak_mb: f64,
    pub vram_allocated_mb: f64,
    pub prompt_tokens: u32,
    pub generated_tokens: u32,
    pub copied_tensor_count: u64,
    pub converted_tensor_count: u64,
    pub converted_bytes_before: u64,
    pub converted_bytes_after: u64,
    pub requested_target_count: u64,
    pub verified_target_count: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct MsRecipeTensorTarget {
    name: *const c_char,
    target_quant: *const c_char,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MsRecipeAnalysis {
    pub tensor_count: u64,
    pub changed_count: u64,
    pub unsupported_count: u64,
    pub missing_count: u64,
    pub unknown_quant_count: u64,
    pub current_size_bytes: u64,
    pub estimated_target_size_bytes: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MsRecipeEvalResult {
    pub baseline_load_ms: f64,
    pub baseline_prompt_eval_ms: f64,
    pub baseline_generation_ms: f64,
    pub baseline_prompt_eval_tps: f64,
    pub baseline_token_gen_tps: f64,
    pub baseline_ttft_ms: f64,
    pub baseline_runtime_elapsed_ms: f64,
    pub baseline_nll: f64,
    pub baseline_ppl: f64,
    pub baseline_eval_ms: f64,
    pub baseline_vram_peak_mb: f64,
    pub baseline_vram_allocated_mb: f64,
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

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct MsStandardEvalSample {
    task: *const c_char,
    prompt: *const c_char,
    choices: *const *const c_char,
    choice_count: u64,
    gold_index: u32,
    normalize_by_choice_length: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MsStandardEvalTaskResult {
    pub task: [c_char; 64],
    pub sample_count: u64,
    pub baseline_correct_count: u64,
    pub recipe_correct_count: u64,
    pub correct_to_wrong_count: u64,
    pub wrong_to_correct_count: u64,
    pub same_prediction_count: u64,
    pub baseline_accuracy: f64,
    pub recipe_accuracy: f64,
    pub accuracy_delta: f64,
    pub baseline_avg_margin: f64,
    pub recipe_avg_margin: f64,
    pub margin_delta: f64,
    pub baseline_avg_correct_nll: f64,
    pub recipe_avg_correct_nll: f64,
}

#[derive(Debug, Clone)]
pub struct StandardEvalSampleInput {
    pub task: String,
    pub prompt: String,
    pub choices: Vec<String>,
    pub gold_index: u32,
    pub normalize_by_choice_length: bool,
}

extern "C" {
    fn ms_runtime_version() -> *const c_char;
    fn ms_runtime_llama_system_info() -> *const c_char;
    fn ms_runtime_last_error() -> *const c_char;
    fn ms_runtime_inspect_gguf(path: *const c_char, out_summary: *mut MsGgufSummary) -> c_int;
    fn ms_runtime_analyze_recipe(
        path: *const c_char,
        targets: *const MsRecipeTensorTarget,
        target_count: u64,
        out_analysis: *mut MsRecipeAnalysis,
    ) -> c_int;
    fn ms_runtime_benchmark_baseline(
        path: *const c_char,
        prompt: *const c_char,
        max_tokens: u32,
        out_benchmark: *mut MsBaselineBenchmark,
    ) -> c_int;
    fn ms_runtime_benchmark_user_copy(
        path: *const c_char,
        prompt: *const c_char,
        max_tokens: u32,
        out_benchmark: *mut MsBaselineBenchmark,
    ) -> c_int;
    fn ms_runtime_benchmark_recipe(
        path: *const c_char,
        targets: *const MsRecipeTensorTarget,
        target_count: u64,
        prompt: *const c_char,
        max_tokens: u32,
        out_benchmark: *mut MsBaselineBenchmark,
    ) -> c_int;
    fn ms_runtime_eval_recipe(
        path: *const c_char,
        targets: *const MsRecipeTensorTarget,
        target_count: u64,
        eval_texts: *const *const c_char,
        eval_text_count: u64,
        max_eval_tokens: u32,
        prompt: *const c_char,
        max_tokens: u32,
        out_benchmark: *mut MsBaselineBenchmark,
        out_eval: *mut MsRecipeEvalResult,
    ) -> c_int;
    fn ms_runtime_eval_recipe_single(
        path: *const c_char,
        targets: *const MsRecipeTensorTarget,
        target_count: u64,
        eval_texts: *const *const c_char,
        eval_text_count: u64,
        max_eval_tokens: u32,
        prompt: *const c_char,
        max_tokens: u32,
        out_benchmark: *mut MsBaselineBenchmark,
        out_eval: *mut MsRecipeEvalResult,
    ) -> c_int;
    fn ms_runtime_eval_recipe_standard(
        path: *const c_char,
        targets: *const MsRecipeTensorTarget,
        target_count: u64,
        eval_texts: *const *const c_char,
        eval_text_count: u64,
        standard_samples: *const MsStandardEvalSample,
        standard_sample_count: u64,
        max_eval_tokens: u32,
        prompt: *const c_char,
        max_tokens: u32,
        out_benchmark: *mut MsBaselineBenchmark,
        out_eval: *mut MsRecipeEvalResult,
        out_task_results: *mut MsStandardEvalTaskResult,
        task_result_capacity: u64,
        out_task_result_count: *mut u64,
    ) -> c_int;
    fn ms_runtime_eval_recipe_standard_single(
        path: *const c_char,
        targets: *const MsRecipeTensorTarget,
        target_count: u64,
        eval_texts: *const *const c_char,
        eval_text_count: u64,
        standard_samples: *const MsStandardEvalSample,
        standard_sample_count: u64,
        max_eval_tokens: u32,
        prompt: *const c_char,
        max_tokens: u32,
        out_benchmark: *mut MsBaselineBenchmark,
        out_eval: *mut MsRecipeEvalResult,
        out_task_results: *mut MsStandardEvalTaskResult,
        task_result_capacity: u64,
        out_task_result_count: *mut u64,
    ) -> c_int;
}

pub fn runtime_version() -> String {
    unsafe { c_string(ms_runtime_version()) }
}

pub fn llama_system_info() -> String {
    unsafe { c_string(ms_runtime_llama_system_info()) }
}

pub fn inspect_gguf(path: &str) -> Result<MsGgufSummary, String> {
    let c_path =
        CString::new(path).map_err(|_| "GGUF path contains an interior NUL byte".to_string())?;
    let mut summary = MsGgufSummary {
        version: 0,
        tensor_count: 0,
        metadata_count: 0,
        alignment: 0,
        data_offset: 0,
    };

    let result = unsafe { ms_runtime_inspect_gguf(c_path.as_ptr(), &mut summary) };
    if result == 0 {
        Ok(summary)
    } else {
        Err(unsafe { c_string(ms_runtime_last_error()) })
    }
}

pub fn analyze_recipe(
    path: &str,
    targets: &[(String, String)],
) -> Result<MsRecipeAnalysis, String> {
    let c_path =
        CString::new(path).map_err(|_| "GGUF path contains an interior NUL byte".to_string())?;
    let c_names = targets
        .iter()
        .map(|(name, _)| {
            CString::new(name.as_str())
                .map_err(|_| format!("tensor name contains an interior NUL byte: {}", name))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let c_quants = targets
        .iter()
        .map(|(_, quant)| {
            CString::new(quant.as_str())
                .map_err(|_| format!("quant type contains an interior NUL byte: {}", quant))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let native_targets = c_names
        .iter()
        .zip(c_quants.iter())
        .map(|(name, quant)| MsRecipeTensorTarget {
            name: name.as_ptr(),
            target_quant: quant.as_ptr(),
        })
        .collect::<Vec<_>>();
    let mut analysis = MsRecipeAnalysis {
        tensor_count: 0,
        changed_count: 0,
        unsupported_count: 0,
        missing_count: 0,
        unknown_quant_count: 0,
        current_size_bytes: 0,
        estimated_target_size_bytes: 0,
    };

    let result = unsafe {
        ms_runtime_analyze_recipe(
            c_path.as_ptr(),
            native_targets.as_ptr(),
            native_targets.len() as u64,
            &mut analysis,
        )
    };
    if result == 0 {
        Ok(analysis)
    } else {
        Err(unsafe { c_string(ms_runtime_last_error()) })
    }
}

pub fn benchmark_baseline(
    path: &str,
    prompt: &str,
    max_tokens: u32,
) -> Result<MsBaselineBenchmark, String> {
    let c_path =
        CString::new(path).map_err(|_| "GGUF path contains an interior NUL byte".to_string())?;
    let c_prompt = CString::new(prompt)
        .map_err(|_| "benchmark prompt contains an interior NUL byte".to_string())?;
    let mut benchmark = MsBaselineBenchmark {
        load_ms: 0.0,
        prompt_eval_ms: 0.0,
        generation_ms: 0.0,
        prompt_eval_tps: 0.0,
        token_gen_tps: 0.0,
        ttft_ms: 0.0,
        vram_peak_mb: 0.0,
        vram_allocated_mb: 0.0,
        prompt_tokens: 0,
        generated_tokens: 0,
        copied_tensor_count: 0,
        converted_tensor_count: 0,
        converted_bytes_before: 0,
        converted_bytes_after: 0,
        requested_target_count: 0,
        verified_target_count: 0,
    };

    let result = unsafe {
        ms_runtime_benchmark_baseline(
            c_path.as_ptr(),
            c_prompt.as_ptr(),
            max_tokens,
            &mut benchmark,
        )
    };
    if result == 0 {
        Ok(benchmark)
    } else {
        Err(unsafe { c_string(ms_runtime_last_error()) })
    }
}

pub fn benchmark_user_copy(
    path: &str,
    prompt: &str,
    max_tokens: u32,
) -> Result<MsBaselineBenchmark, String> {
    let c_path =
        CString::new(path).map_err(|_| "GGUF path contains an interior NUL byte".to_string())?;
    let c_prompt = CString::new(prompt)
        .map_err(|_| "benchmark prompt contains an interior NUL byte".to_string())?;
    let mut benchmark = MsBaselineBenchmark {
        load_ms: 0.0,
        prompt_eval_ms: 0.0,
        generation_ms: 0.0,
        prompt_eval_tps: 0.0,
        token_gen_tps: 0.0,
        ttft_ms: 0.0,
        vram_peak_mb: 0.0,
        vram_allocated_mb: 0.0,
        prompt_tokens: 0,
        generated_tokens: 0,
        copied_tensor_count: 0,
        converted_tensor_count: 0,
        converted_bytes_before: 0,
        converted_bytes_after: 0,
        requested_target_count: 0,
        verified_target_count: 0,
    };

    let result = unsafe {
        ms_runtime_benchmark_user_copy(
            c_path.as_ptr(),
            c_prompt.as_ptr(),
            max_tokens,
            &mut benchmark,
        )
    };
    if result == 0 {
        Ok(benchmark)
    } else {
        Err(unsafe { c_string(ms_runtime_last_error()) })
    }
}

pub fn benchmark_recipe(
    path: &str,
    targets: &[(String, String)],
    prompt: &str,
    max_tokens: u32,
) -> Result<MsBaselineBenchmark, String> {
    let c_path =
        CString::new(path).map_err(|_| "GGUF path contains an interior NUL byte".to_string())?;
    let c_prompt = CString::new(prompt)
        .map_err(|_| "benchmark prompt contains an interior NUL byte".to_string())?;
    let c_names = targets
        .iter()
        .map(|(name, _)| {
            CString::new(name.as_str())
                .map_err(|_| format!("tensor name contains an interior NUL byte: {}", name))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let c_quants = targets
        .iter()
        .map(|(_, quant)| {
            CString::new(quant.as_str())
                .map_err(|_| format!("quant type contains an interior NUL byte: {}", quant))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let native_targets = c_names
        .iter()
        .zip(c_quants.iter())
        .map(|(name, quant)| MsRecipeTensorTarget {
            name: name.as_ptr(),
            target_quant: quant.as_ptr(),
        })
        .collect::<Vec<_>>();
    let mut benchmark = MsBaselineBenchmark {
        load_ms: 0.0,
        prompt_eval_ms: 0.0,
        generation_ms: 0.0,
        prompt_eval_tps: 0.0,
        token_gen_tps: 0.0,
        ttft_ms: 0.0,
        vram_peak_mb: 0.0,
        vram_allocated_mb: 0.0,
        prompt_tokens: 0,
        generated_tokens: 0,
        copied_tensor_count: 0,
        converted_tensor_count: 0,
        converted_bytes_before: 0,
        converted_bytes_after: 0,
        requested_target_count: 0,
        verified_target_count: 0,
    };

    let result = unsafe {
        ms_runtime_benchmark_recipe(
            c_path.as_ptr(),
            native_targets.as_ptr(),
            native_targets.len() as u64,
            c_prompt.as_ptr(),
            max_tokens,
            &mut benchmark,
        )
    };
    if result == 0 {
        Ok(benchmark)
    } else {
        Err(unsafe { c_string(ms_runtime_last_error()) })
    }
}

pub fn eval_recipe(
    path: &str,
    targets: &[(String, String)],
    eval_texts: &[String],
    max_eval_tokens: u32,
    prompt: &str,
    max_tokens: u32,
) -> Result<(MsBaselineBenchmark, MsRecipeEvalResult), String> {
    eval_recipe_with_native_fn(
        path,
        targets,
        eval_texts,
        max_eval_tokens,
        prompt,
        max_tokens,
        ms_runtime_eval_recipe,
    )
}

pub fn eval_recipe_single(
    path: &str,
    targets: &[(String, String)],
    eval_texts: &[String],
    max_eval_tokens: u32,
    prompt: &str,
    max_tokens: u32,
) -> Result<(MsBaselineBenchmark, MsRecipeEvalResult), String> {
    eval_recipe_with_native_fn(
        path,
        targets,
        eval_texts,
        max_eval_tokens,
        prompt,
        max_tokens,
        ms_runtime_eval_recipe_single,
    )
}

pub fn eval_recipe_standard(
    path: &str,
    targets: &[(String, String)],
    eval_texts: &[String],
    standard_samples: &[StandardEvalSampleInput],
    max_eval_tokens: u32,
    prompt: &str,
    max_tokens: u32,
) -> Result<
    (
        MsBaselineBenchmark,
        MsRecipeEvalResult,
        Vec<MsStandardEvalTaskResult>,
    ),
    String,
> {
    eval_recipe_standard_with_native_fn(
        path,
        targets,
        eval_texts,
        standard_samples,
        max_eval_tokens,
        prompt,
        max_tokens,
        ms_runtime_eval_recipe_standard,
    )
}

pub fn eval_recipe_standard_single(
    path: &str,
    targets: &[(String, String)],
    eval_texts: &[String],
    standard_samples: &[StandardEvalSampleInput],
    max_eval_tokens: u32,
    prompt: &str,
    max_tokens: u32,
) -> Result<
    (
        MsBaselineBenchmark,
        MsRecipeEvalResult,
        Vec<MsStandardEvalTaskResult>,
    ),
    String,
> {
    eval_recipe_standard_with_native_fn(
        path,
        targets,
        eval_texts,
        standard_samples,
        max_eval_tokens,
        prompt,
        max_tokens,
        ms_runtime_eval_recipe_standard_single,
    )
}

fn eval_recipe_with_native_fn(
    path: &str,
    targets: &[(String, String)],
    eval_texts: &[String],
    max_eval_tokens: u32,
    prompt: &str,
    max_tokens: u32,
    native_fn: unsafe extern "C" fn(
        *const c_char,
        *const MsRecipeTensorTarget,
        u64,
        *const *const c_char,
        u64,
        u32,
        *const c_char,
        u32,
        *mut MsBaselineBenchmark,
        *mut MsRecipeEvalResult,
    ) -> c_int,
) -> Result<(MsBaselineBenchmark, MsRecipeEvalResult), String> {
    let c_path =
        CString::new(path).map_err(|_| "GGUF path contains an interior NUL byte".to_string())?;
    let c_prompt = CString::new(prompt)
        .map_err(|_| "benchmark prompt contains an interior NUL byte".to_string())?;
    let c_names = targets
        .iter()
        .map(|(name, _)| {
            CString::new(name.as_str())
                .map_err(|_| format!("tensor name contains an interior NUL byte: {}", name))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let c_quants = targets
        .iter()
        .map(|(_, quant)| {
            CString::new(quant.as_str())
                .map_err(|_| format!("quant type contains an interior NUL byte: {}", quant))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let c_eval_texts = eval_texts
        .iter()
        .map(|text| {
            CString::new(text.as_str())
                .map_err(|_| "eval text contains an interior NUL byte".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let eval_text_ptrs = c_eval_texts
        .iter()
        .map(|text| text.as_ptr())
        .collect::<Vec<_>>();
    let native_targets = c_names
        .iter()
        .zip(c_quants.iter())
        .map(|(name, quant)| MsRecipeTensorTarget {
            name: name.as_ptr(),
            target_quant: quant.as_ptr(),
        })
        .collect::<Vec<_>>();
    let mut benchmark = MsBaselineBenchmark {
        load_ms: 0.0,
        prompt_eval_ms: 0.0,
        generation_ms: 0.0,
        prompt_eval_tps: 0.0,
        token_gen_tps: 0.0,
        ttft_ms: 0.0,
        vram_peak_mb: 0.0,
        vram_allocated_mb: 0.0,
        prompt_tokens: 0,
        generated_tokens: 0,
        copied_tensor_count: 0,
        converted_tensor_count: 0,
        converted_bytes_before: 0,
        converted_bytes_after: 0,
        requested_target_count: 0,
        verified_target_count: 0,
    };
    let mut eval = MsRecipeEvalResult {
        baseline_load_ms: 0.0,
        baseline_prompt_eval_ms: 0.0,
        baseline_generation_ms: 0.0,
        baseline_prompt_eval_tps: 0.0,
        baseline_token_gen_tps: 0.0,
        baseline_ttft_ms: 0.0,
        baseline_runtime_elapsed_ms: 0.0,
        baseline_nll: 0.0,
        baseline_ppl: 0.0,
        baseline_eval_ms: 0.0,
        baseline_vram_peak_mb: 0.0,
        baseline_vram_allocated_mb: 0.0,
        recipe_nll: 0.0,
        recipe_ppl: 0.0,
        recipe_eval_ms: 0.0,
        recipe_vram_peak_mb: 0.0,
        recipe_vram_allocated_mb: 0.0,
        ppl_delta: 0.0,
        ppl_delta_percent: 0.0,
        eval_token_count: 0,
        eval_sample_count: 0,
        skipped_sample_count: 0,
    };

    let result = unsafe {
        native_fn(
            c_path.as_ptr(),
            native_targets.as_ptr(),
            native_targets.len() as u64,
            eval_text_ptrs.as_ptr(),
            eval_text_ptrs.len() as u64,
            max_eval_tokens,
            c_prompt.as_ptr(),
            max_tokens,
            &mut benchmark,
            &mut eval,
        )
    };
    if result == 0 {
        Ok((benchmark, eval))
    } else {
        Err(unsafe { c_string(ms_runtime_last_error()) })
    }
}

fn eval_recipe_standard_with_native_fn(
    path: &str,
    targets: &[(String, String)],
    eval_texts: &[String],
    standard_samples: &[StandardEvalSampleInput],
    max_eval_tokens: u32,
    prompt: &str,
    max_tokens: u32,
    native_fn: unsafe extern "C" fn(
        *const c_char,
        *const MsRecipeTensorTarget,
        u64,
        *const *const c_char,
        u64,
        *const MsStandardEvalSample,
        u64,
        u32,
        *const c_char,
        u32,
        *mut MsBaselineBenchmark,
        *mut MsRecipeEvalResult,
        *mut MsStandardEvalTaskResult,
        u64,
        *mut u64,
    ) -> c_int,
) -> Result<
    (
        MsBaselineBenchmark,
        MsRecipeEvalResult,
        Vec<MsStandardEvalTaskResult>,
    ),
    String,
> {
    let c_path =
        CString::new(path).map_err(|_| "GGUF path contains an interior NUL byte".to_string())?;
    let c_prompt = CString::new(prompt)
        .map_err(|_| "benchmark prompt contains an interior NUL byte".to_string())?;
    let c_names = targets
        .iter()
        .map(|(name, _)| {
            CString::new(name.as_str())
                .map_err(|_| format!("tensor name contains an interior NUL byte: {}", name))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let c_quants = targets
        .iter()
        .map(|(_, quant)| {
            CString::new(quant.as_str())
                .map_err(|_| format!("quant type contains an interior NUL byte: {}", quant))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let c_eval_texts = eval_texts
        .iter()
        .map(|text| {
            CString::new(text.as_str())
                .map_err(|_| "eval text contains an interior NUL byte".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let eval_text_ptrs = c_eval_texts
        .iter()
        .map(|text| text.as_ptr())
        .collect::<Vec<_>>();
    let native_targets = c_names
        .iter()
        .zip(c_quants.iter())
        .map(|(name, quant)| MsRecipeTensorTarget {
            name: name.as_ptr(),
            target_quant: quant.as_ptr(),
        })
        .collect::<Vec<_>>();

    let c_standard_tasks = standard_samples
        .iter()
        .map(|sample| {
            CString::new(sample.task.as_str()).map_err(|_| {
                format!(
                    "standard eval task contains an interior NUL byte: {}",
                    sample.task
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let c_standard_prompts = standard_samples
        .iter()
        .map(|sample| {
            CString::new(sample.prompt.as_str()).map_err(|_| {
                format!(
                    "standard eval prompt contains an interior NUL byte: {}",
                    sample.task
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let c_standard_choices = standard_samples
        .iter()
        .map(|sample| {
            sample
                .choices
                .iter()
                .map(|choice| {
                    CString::new(choice.as_str()).map_err(|_| {
                        format!(
                            "standard eval choice contains an interior NUL byte: {}",
                            sample.task
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let standard_choice_ptrs = c_standard_choices
        .iter()
        .map(|choices| {
            choices
                .iter()
                .map(|choice| choice.as_ptr())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let native_standard_samples = standard_samples
        .iter()
        .enumerate()
        .map(|(index, sample)| MsStandardEvalSample {
            task: c_standard_tasks[index].as_ptr(),
            prompt: c_standard_prompts[index].as_ptr(),
            choices: standard_choice_ptrs[index].as_ptr(),
            choice_count: standard_choice_ptrs[index].len() as u64,
            gold_index: sample.gold_index,
            normalize_by_choice_length: u32::from(sample.normalize_by_choice_length),
        })
        .collect::<Vec<_>>();

    let mut benchmark = empty_benchmark();
    let mut eval = empty_eval();
    let mut task_results = vec![empty_standard_task_result(); standard_samples.len().max(1)];
    let mut task_result_count = 0u64;

    let result = unsafe {
        native_fn(
            c_path.as_ptr(),
            native_targets.as_ptr(),
            native_targets.len() as u64,
            eval_text_ptrs.as_ptr(),
            eval_text_ptrs.len() as u64,
            native_standard_samples.as_ptr(),
            native_standard_samples.len() as u64,
            max_eval_tokens,
            c_prompt.as_ptr(),
            max_tokens,
            &mut benchmark,
            &mut eval,
            task_results.as_mut_ptr(),
            task_results.len() as u64,
            &mut task_result_count,
        )
    };
    if result == 0 {
        task_results.truncate(task_result_count as usize);
        Ok((benchmark, eval, task_results))
    } else {
        Err(unsafe { c_string(ms_runtime_last_error()) })
    }
}

fn empty_benchmark() -> MsBaselineBenchmark {
    MsBaselineBenchmark {
        load_ms: 0.0,
        prompt_eval_ms: 0.0,
        generation_ms: 0.0,
        prompt_eval_tps: 0.0,
        token_gen_tps: 0.0,
        ttft_ms: 0.0,
        vram_peak_mb: 0.0,
        vram_allocated_mb: 0.0,
        prompt_tokens: 0,
        generated_tokens: 0,
        copied_tensor_count: 0,
        converted_tensor_count: 0,
        converted_bytes_before: 0,
        converted_bytes_after: 0,
        requested_target_count: 0,
        verified_target_count: 0,
    }
}

fn empty_eval() -> MsRecipeEvalResult {
    MsRecipeEvalResult {
        baseline_load_ms: 0.0,
        baseline_prompt_eval_ms: 0.0,
        baseline_generation_ms: 0.0,
        baseline_prompt_eval_tps: 0.0,
        baseline_token_gen_tps: 0.0,
        baseline_ttft_ms: 0.0,
        baseline_runtime_elapsed_ms: 0.0,
        baseline_nll: 0.0,
        baseline_ppl: 0.0,
        baseline_eval_ms: 0.0,
        baseline_vram_peak_mb: 0.0,
        baseline_vram_allocated_mb: 0.0,
        recipe_nll: 0.0,
        recipe_ppl: 0.0,
        recipe_eval_ms: 0.0,
        recipe_vram_peak_mb: 0.0,
        recipe_vram_allocated_mb: 0.0,
        ppl_delta: 0.0,
        ppl_delta_percent: 0.0,
        eval_token_count: 0,
        eval_sample_count: 0,
        skipped_sample_count: 0,
    }
}

fn empty_standard_task_result() -> MsStandardEvalTaskResult {
    MsStandardEvalTaskResult {
        task: [0; 64],
        sample_count: 0,
        baseline_correct_count: 0,
        recipe_correct_count: 0,
        correct_to_wrong_count: 0,
        wrong_to_correct_count: 0,
        same_prediction_count: 0,
        baseline_accuracy: 0.0,
        recipe_accuracy: 0.0,
        accuracy_delta: 0.0,
        baseline_avg_margin: 0.0,
        recipe_avg_margin: 0.0,
        margin_delta: 0.0,
        baseline_avg_correct_nll: 0.0,
        recipe_avg_correct_nll: 0.0,
    }
}

pub fn standard_task_name(task: &MsStandardEvalTaskResult) -> String {
    let end = task
        .task
        .iter()
        .position(|&c| c == 0)
        .unwrap_or(task.task.len());
    let bytes = task.task[..end]
        .iter()
        .map(|&c| c as u8)
        .collect::<Vec<_>>();
    String::from_utf8_lossy(&bytes).into_owned()
}

unsafe fn c_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    CStr::from_ptr(ptr).to_string_lossy().into_owned()
}
