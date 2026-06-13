#![allow(dead_code)]

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::ptr::NonNull;

const STANDARD_EVAL_AUDIT_MAX_CHOICES: usize = 32;

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
struct MsChatMessage {
    role: *const c_char,
    content: *const c_char,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChatGenerationParams {
    pub max_tokens: u32,
    pub add_generation_prompt: u32,
    pub seed: u32,
    pub top_k: i32,
    pub repeat_last_n: i32,
    pub dry_allowed_length: i32,
    pub dry_penalty_last_n: i32,
    pub temperature: f64,
    pub top_p: f64,
    pub min_p: f64,
    pub typical_p: f64,
    pub repeat_penalty: f64,
    pub frequency_penalty: f64,
    pub presence_penalty: f64,
    pub dry_multiplier: f64,
    pub dry_base: f64,
}

impl Default for ChatGenerationParams {
    fn default() -> Self {
        Self {
            max_tokens: 1024,
            add_generation_prompt: 1,
            seed: u32::MAX,
            top_k: 40,
            repeat_last_n: 64,
            dry_allowed_length: 2,
            dry_penalty_last_n: -1,
            temperature: 0.8,
            top_p: 0.95,
            min_p: 0.05,
            typical_p: 1.0,
            repeat_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            dry_multiplier: 0.0,
            dry_base: 1.75,
        }
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatFinishReason {
    Stop = 0,
    Length = 1,
    Eos = 2,
}

impl ChatFinishReason {
    fn from_native(value: u32) -> Result<Self, String> {
        match value {
            0 => Ok(Self::Stop),
            1 => Ok(Self::Length),
            2 => Ok(Self::Eos),
            _ => Err(format!("unknown native chat finish reason: {value}")),
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct MsChatGenerationResult {
    benchmark: MsBaselineBenchmark,
    prompt_tokens: u32,
    completion_tokens: u32,
    finish_reason: u32,
}

#[derive(Debug, Clone)]
pub struct ChatGenerationOutput {
    pub text: String,
    pub reasoning_text: Option<String>,
    pub benchmark: MsBaselineBenchmark,
    pub finish_reason: ChatFinishReason,
}

#[repr(C)]
struct MsRuntimeChatSession {
    _private: [u8; 0],
}

type MsRuntimeLogCallback = Option<unsafe extern "C" fn(*const c_char, *mut c_void)>;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MsRuntimeChatSessionCounters {
    pub model_load_count: u64,
    pub context_reset_count: u64,
    pub completion_count: u64,
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
    pub baseline_ppl_uncertainty: f64,
    pub baseline_eval_ms: f64,
    pub baseline_vram_peak_mb: f64,
    pub baseline_vram_allocated_mb: f64,
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

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct MsStandardEvalSample {
    task: *const c_char,
    prompt: *const c_char,
    choices: *const *const c_char,
    choice_lengths: *const u64,
    choice_count: u64,
    gold_index: u32,
    normalize_by_choice_length: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MsStandardEvalSampleAudit {
    pub sample_index: u64,
    pub task: [c_char; 64],
    pub choice_count: u32,
    pub gold_index: u32,
    pub has_baseline: u32,
    pub baseline_prediction_index: u32,
    pub recipe_prediction_index: u32,
    pub baseline_correct: u32,
    pub recipe_correct: u32,
    pub choice_denominators: [f64; STANDARD_EVAL_AUDIT_MAX_CHOICES],
    pub baseline_choice_nlls: [f64; STANDARD_EVAL_AUDIT_MAX_CHOICES],
    pub baseline_choice_scores: [f64; STANDARD_EVAL_AUDIT_MAX_CHOICES],
    pub recipe_choice_nlls: [f64; STANDARD_EVAL_AUDIT_MAX_CHOICES],
    pub recipe_choice_scores: [f64; STANDARD_EVAL_AUDIT_MAX_CHOICES],
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
    pub doc_id: String,
    pub task: String,
    pub prompt: String,
    pub target_delimiter: String,
    pub choices: Vec<String>,
    pub continuations: Vec<String>,
    pub choice_lengths: Vec<u64>,
    pub gold_index: u32,
    pub normalize_by_choice_length: bool,
}

extern "C" {
    fn ms_runtime_version() -> *const c_char;
    fn ms_runtime_llama_system_info() -> *const c_char;
    fn ms_runtime_last_error() -> *const c_char;
    fn ms_runtime_reset_recipe_test_cancel();
    fn ms_runtime_cancel_recipe_test();
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
    fn ms_runtime_generate_recipe(
        path: *const c_char,
        targets: *const MsRecipeTensorTarget,
        target_count: u64,
        prompt: *const c_char,
        max_tokens: u32,
        out_text: *mut c_char,
        out_text_capacity: u64,
        out_benchmark: *mut MsBaselineBenchmark,
    ) -> c_int;
    fn ms_runtime_generate_recipe_chat(
        path: *const c_char,
        targets: *const MsRecipeTensorTarget,
        target_count: u64,
        messages: *const MsChatMessage,
        message_count: u64,
        max_tokens: u32,
        out_text: *mut c_char,
        out_text_capacity: u64,
        out_benchmark: *mut MsBaselineBenchmark,
    ) -> c_int;
    fn ms_runtime_open_recipe_chat_session(
        path: *const c_char,
        targets: *const MsRecipeTensorTarget,
        target_count: u64,
        max_tokens: u32,
        out_session: *mut *mut MsRuntimeChatSession,
    ) -> c_int;
    fn ms_runtime_open_recipe_chat_session_with_progress(
        path: *const c_char,
        targets: *const MsRecipeTensorTarget,
        target_count: u64,
        max_tokens: u32,
        log_callback: MsRuntimeLogCallback,
        log_user_data: *mut c_void,
        out_session: *mut *mut MsRuntimeChatSession,
    ) -> c_int;
    fn ms_runtime_close_recipe_chat_session(session: *mut MsRuntimeChatSession);
    fn ms_runtime_generate_recipe_chat_session(
        session: *mut MsRuntimeChatSession,
        messages: *const MsChatMessage,
        message_count: u64,
        params: *const ChatGenerationParams,
        stop_strings: *const *const c_char,
        stop_count: u64,
        chat_template_kwargs_json: *const c_char,
        reasoning_format: *const c_char,
        out_text: *mut c_char,
        out_text_capacity: u64,
        out_reasoning_text: *mut c_char,
        out_reasoning_text_capacity: u64,
        out_result: *mut MsChatGenerationResult,
    ) -> c_int;
    fn ms_runtime_get_recipe_chat_session_counters(
        session: *const MsRuntimeChatSession,
        out_counters: *mut MsRuntimeChatSessionCounters,
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
        out_sample_audits: *mut MsStandardEvalSampleAudit,
        sample_audit_capacity: u64,
        out_sample_audit_count: *mut u64,
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
        out_sample_audits: *mut MsStandardEvalSampleAudit,
        sample_audit_capacity: u64,
        out_sample_audit_count: *mut u64,
    ) -> c_int;
}

pub fn runtime_version() -> String {
    unsafe { c_string(ms_runtime_version()) }
}

pub fn llama_system_info() -> String {
    unsafe { c_string(ms_runtime_llama_system_info()) }
}

pub fn reset_recipe_test_cancel() {
    unsafe { ms_runtime_reset_recipe_test_cancel() }
}

pub fn cancel_recipe_test() {
    unsafe { ms_runtime_cancel_recipe_test() }
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

pub fn generate_recipe(
    path: &str,
    targets: &[(String, String)],
    prompt: &str,
    max_tokens: u32,
) -> Result<(String, MsBaselineBenchmark), String> {
    let c_path =
        CString::new(path).map_err(|_| "GGUF path contains an interior NUL byte".to_string())?;
    let c_prompt = CString::new(prompt)
        .map_err(|_| "generation prompt contains an interior NUL byte".to_string())?;
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
    let mut text_buffer = vec![0 as c_char; 131_072];
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
        ms_runtime_generate_recipe(
            c_path.as_ptr(),
            native_targets.as_ptr(),
            native_targets.len() as u64,
            c_prompt.as_ptr(),
            max_tokens,
            text_buffer.as_mut_ptr(),
            text_buffer.len() as u64,
            &mut benchmark,
        )
    };
    if result == 0 {
        let text = unsafe { CStr::from_ptr(text_buffer.as_ptr()) }
            .to_string_lossy()
            .to_string();
        Ok((text, benchmark))
    } else {
        Err(unsafe { c_string(ms_runtime_last_error()) })
    }
}

pub fn generate_recipe_chat(
    path: &str,
    targets: &[(String, String)],
    messages: &[(String, String)],
    max_tokens: u32,
) -> Result<(String, MsBaselineBenchmark), String> {
    let c_path =
        CString::new(path).map_err(|_| format!("path contains an interior NUL byte: {}", path))?;
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
    let c_roles = messages
        .iter()
        .map(|(role, _)| {
            CString::new(role.as_str())
                .map_err(|_| format!("chat role contains an interior NUL byte: {}", role))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let c_contents = messages
        .iter()
        .map(|(_, content)| {
            CString::new(content.as_str())
                .map_err(|_| "chat message content contains an interior NUL byte".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let native_messages = c_roles
        .iter()
        .zip(c_contents.iter())
        .map(|(role, content)| MsChatMessage {
            role: role.as_ptr(),
            content: content.as_ptr(),
        })
        .collect::<Vec<_>>();
    let mut text_buffer = vec![0 as c_char; 131_072];
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
        ms_runtime_generate_recipe_chat(
            c_path.as_ptr(),
            native_targets.as_ptr(),
            native_targets.len() as u64,
            native_messages.as_ptr(),
            native_messages.len() as u64,
            max_tokens,
            text_buffer.as_mut_ptr(),
            text_buffer.len() as u64,
            &mut benchmark,
        )
    };
    if result == 0 {
        let text = unsafe { CStr::from_ptr(text_buffer.as_ptr()) }
            .to_string_lossy()
            .to_string();
        Ok((text, benchmark))
    } else {
        Err(unsafe { c_string(ms_runtime_last_error()) })
    }
}

#[derive(Debug)]
pub struct RecipeChatSession {
    ptr: NonNull<MsRuntimeChatSession>,
}

unsafe impl Send for RecipeChatSession {}

impl Drop for RecipeChatSession {
    fn drop(&mut self) {
        unsafe { ms_runtime_close_recipe_chat_session(self.ptr.as_ptr()) }
    }
}

impl RecipeChatSession {
    pub fn generate_chat(
        &mut self,
        messages: &[(String, String)],
        params: &ChatGenerationParams,
        stop_strings: &[String],
        chat_template_kwargs_json: Option<&str>,
        reasoning_format: Option<&str>,
    ) -> Result<ChatGenerationOutput, String> {
        let c_roles = messages
            .iter()
            .map(|(role, _)| {
                CString::new(role.as_str())
                    .map_err(|_| format!("chat role contains an interior NUL byte: {}", role))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let c_contents = messages
            .iter()
            .map(|(_, content)| {
                CString::new(content.as_str())
                    .map_err(|_| "chat message content contains an interior NUL byte".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let native_messages = c_roles
            .iter()
            .zip(c_contents.iter())
            .map(|(role, content)| MsChatMessage {
                role: role.as_ptr(),
                content: content.as_ptr(),
            })
            .collect::<Vec<_>>();
        let c_stop_strings = stop_strings
            .iter()
            .map(|stop| {
                CString::new(stop.as_str())
                    .map_err(|_| "chat stop string contains an interior NUL byte".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let native_stop_strings = c_stop_strings
            .iter()
            .map(|stop| stop.as_ptr())
            .collect::<Vec<_>>();
        let c_chat_template_kwargs_json = chat_template_kwargs_json
            .map(|value| {
                CString::new(value)
                    .map_err(|_| "chat_template_kwargs contains an interior NUL byte".to_string())
            })
            .transpose()?;
        let c_reasoning_format = reasoning_format
            .map(|value| {
                CString::new(value)
                    .map_err(|_| "reasoning_format contains an interior NUL byte".to_string())
            })
            .transpose()?;
        let mut text_buffer = vec![0 as c_char; 131_072];
        let mut reasoning_buffer = vec![0 as c_char; 131_072];
        let mut native_result = MsChatGenerationResult {
            benchmark: empty_benchmark(),
            prompt_tokens: 0,
            completion_tokens: 0,
            finish_reason: ChatFinishReason::Stop as u32,
        };

        let status = unsafe {
            ms_runtime_generate_recipe_chat_session(
                self.ptr.as_ptr(),
                native_messages.as_ptr(),
                native_messages.len() as u64,
                params,
                native_stop_strings.as_ptr(),
                native_stop_strings.len() as u64,
                c_chat_template_kwargs_json
                    .as_ref()
                    .map(|value| value.as_ptr())
                    .unwrap_or(std::ptr::null()),
                c_reasoning_format
                    .as_ref()
                    .map(|value| value.as_ptr())
                    .unwrap_or(std::ptr::null()),
                text_buffer.as_mut_ptr(),
                text_buffer.len() as u64,
                reasoning_buffer.as_mut_ptr(),
                reasoning_buffer.len() as u64,
                &mut native_result,
            )
        };
        if status == 0 {
            let text = unsafe { CStr::from_ptr(text_buffer.as_ptr()) }
                .to_string_lossy()
                .to_string();
            let reasoning_text = unsafe { CStr::from_ptr(reasoning_buffer.as_ptr()) }
                .to_string_lossy()
                .to_string();
            Ok(ChatGenerationOutput {
                text,
                reasoning_text: (!reasoning_text.is_empty()).then_some(reasoning_text),
                benchmark: native_result.benchmark,
                finish_reason: ChatFinishReason::from_native(native_result.finish_reason)?,
            })
        } else {
            Err(unsafe { c_string(ms_runtime_last_error()) })
        }
    }

    pub fn counters(&self) -> Result<MsRuntimeChatSessionCounters, String> {
        let mut counters = MsRuntimeChatSessionCounters {
            model_load_count: 0,
            context_reset_count: 0,
            completion_count: 0,
        };
        let result = unsafe {
            ms_runtime_get_recipe_chat_session_counters(self.ptr.as_ptr(), &mut counters)
        };
        if result == 0 {
            Ok(counters)
        } else {
            Err(unsafe { c_string(ms_runtime_last_error()) })
        }
    }
}

pub fn open_recipe_chat_session(
    path: &str,
    targets: &[(String, String)],
    max_tokens: u32,
) -> Result<RecipeChatSession, String> {
    open_recipe_chat_session_with_native_call(
        path,
        targets,
        max_tokens,
        |c_path, native_targets, session| unsafe {
            ms_runtime_open_recipe_chat_session(
                c_path,
                native_targets.as_ptr(),
                native_targets.len() as u64,
                max_tokens,
                session,
            )
        },
    )
}

pub fn open_recipe_chat_session_with_progress<F>(
    path: &str,
    targets: &[(String, String)],
    max_tokens: u32,
    mut on_log: F,
) -> Result<RecipeChatSession, String>
where
    F: FnMut(&str),
{
    open_recipe_chat_session_with_native_call(
        path,
        targets,
        max_tokens,
        |c_path, native_targets, session| {
            let log_user_data = &mut on_log as *mut F as *mut c_void;
            unsafe {
                ms_runtime_open_recipe_chat_session_with_progress(
                    c_path,
                    native_targets.as_ptr(),
                    native_targets.len() as u64,
                    max_tokens,
                    Some(recipe_chat_session_log_trampoline::<F>),
                    log_user_data,
                    session,
                )
            }
        },
    )
}

unsafe extern "C" fn recipe_chat_session_log_trampoline<F>(
    message: *const c_char,
    user_data: *mut c_void,
) where
    F: FnMut(&str),
{
    if message.is_null() || user_data.is_null() {
        return;
    }
    let callback = unsafe { &mut *(user_data as *mut F) };
    let message = unsafe { CStr::from_ptr(message) }.to_string_lossy();
    callback(&message);
}

fn open_recipe_chat_session_with_native_call<F>(
    path: &str,
    targets: &[(String, String)],
    max_tokens: u32,
    open: F,
) -> Result<RecipeChatSession, String>
where
    F: FnOnce(*const c_char, &[MsRecipeTensorTarget], *mut *mut MsRuntimeChatSession) -> c_int,
{
    let c_path =
        CString::new(path).map_err(|_| format!("path contains an interior NUL byte: {}", path))?;
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
    let mut session = std::ptr::null_mut();

    let _ = max_tokens;
    let result = open(c_path.as_ptr(), &native_targets, &mut session);
    if result == 0 {
        NonNull::new(session)
            .map(|ptr| RecipeChatSession { ptr })
            .ok_or_else(|| "native recipe chat session pointer is null".to_string())
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
        Vec<MsStandardEvalSampleAudit>,
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
        Vec<MsStandardEvalSampleAudit>,
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
        baseline_ppl_uncertainty: 0.0,
        baseline_eval_ms: 0.0,
        baseline_vram_peak_mb: 0.0,
        baseline_vram_allocated_mb: 0.0,
        recipe_nll: 0.0,
        recipe_ppl: 0.0,
        recipe_ppl_uncertainty: 0.0,
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
        *mut MsStandardEvalSampleAudit,
        u64,
        *mut u64,
    ) -> c_int,
) -> Result<
    (
        MsBaselineBenchmark,
        MsRecipeEvalResult,
        Vec<MsStandardEvalTaskResult>,
        Vec<MsStandardEvalSampleAudit>,
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
                .continuations
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
    for sample in standard_samples {
        if sample.continuations.len() != sample.choice_lengths.len() {
            return Err(format!(
                "standard eval choice length count does not match continuation count: {}",
                sample.task
            ));
        }
        if sample.choice_lengths.iter().any(|length| *length == 0) {
            return Err(format!(
                "standard eval choice length must be positive: {}",
                sample.task
            ));
        }
    }
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
            choice_lengths: sample.choice_lengths.as_ptr(),
            choice_count: standard_choice_ptrs[index].len() as u64,
            gold_index: sample.gold_index,
            normalize_by_choice_length: u32::from(sample.normalize_by_choice_length),
        })
        .collect::<Vec<_>>();

    let mut benchmark = empty_benchmark();
    let mut eval = empty_eval();
    let mut task_results = vec![empty_standard_task_result(); standard_samples.len().max(1)];
    let mut task_result_count = 0u64;
    let mut sample_audits =
        vec![empty_standard_sample_audit(); 20usize.min(standard_samples.len().max(1))];
    let mut sample_audit_count = 0u64;

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
            sample_audits.as_mut_ptr(),
            sample_audits.len() as u64,
            &mut sample_audit_count,
        )
    };
    if result == 0 {
        task_results.truncate(task_result_count as usize);
        sample_audits.truncate(sample_audit_count as usize);
        Ok((benchmark, eval, task_results, sample_audits))
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
        baseline_ppl_uncertainty: 0.0,
        baseline_eval_ms: 0.0,
        baseline_vram_peak_mb: 0.0,
        baseline_vram_allocated_mb: 0.0,
        recipe_nll: 0.0,
        recipe_ppl: 0.0,
        recipe_ppl_uncertainty: 0.0,
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

fn empty_standard_sample_audit() -> MsStandardEvalSampleAudit {
    MsStandardEvalSampleAudit {
        sample_index: 0,
        task: [0; 64],
        choice_count: 0,
        gold_index: 0,
        has_baseline: 0,
        baseline_prediction_index: 0,
        recipe_prediction_index: 0,
        baseline_correct: 0,
        recipe_correct: 0,
        choice_denominators: [0.0; STANDARD_EVAL_AUDIT_MAX_CHOICES],
        baseline_choice_nlls: [0.0; STANDARD_EVAL_AUDIT_MAX_CHOICES],
        baseline_choice_scores: [0.0; STANDARD_EVAL_AUDIT_MAX_CHOICES],
        recipe_choice_nlls: [0.0; STANDARD_EVAL_AUDIT_MAX_CHOICES],
        recipe_choice_scores: [0.0; STANDARD_EVAL_AUDIT_MAX_CHOICES],
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
