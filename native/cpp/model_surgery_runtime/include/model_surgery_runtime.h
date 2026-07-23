#pragma once

#include <stdint.h>

#if defined(_WIN32)
#  if defined(MODEL_SURGERY_RUNTIME_BUILD)
#    define MS_RUNTIME_API __declspec(dllexport)
#  else
#    define MS_RUNTIME_API __declspec(dllimport)
#  endif
#else
#  define MS_RUNTIME_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ms_gguf_summary {
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_count;
    uint64_t alignment;
    uint64_t data_offset;
} ms_gguf_summary;

typedef struct ms_baseline_benchmark {
    double load_ms;
    double prompt_eval_ms;
    double generation_ms;
    double prompt_eval_tps;
    double token_gen_tps;
    double ttft_ms;
    double vram_peak_mb;
    double vram_allocated_mb;
    uint32_t prompt_tokens;
    uint32_t generated_tokens;
    uint64_t copied_tensor_count;
    uint64_t converted_tensor_count;
    uint64_t converted_bytes_before;
    uint64_t converted_bytes_after;
    uint64_t requested_target_count;
    uint64_t verified_target_count;
} ms_baseline_benchmark;

typedef struct ms_recipe_tensor_target {
    const char * name;
    const char * target_quant;
} ms_recipe_tensor_target;

typedef struct ms_chat_message {
    const char * role;
    const char * content;
} ms_chat_message;

typedef struct ms_chat_generation_params {
    uint32_t max_tokens;
    uint32_t add_generation_prompt;
    uint32_t seed;
    int32_t top_k;
    int32_t repeat_last_n;
    int32_t dry_allowed_length;
    int32_t dry_penalty_last_n;
    double temperature;
    double top_p;
    double min_p;
    double typical_p;
    double repeat_penalty;
    double frequency_penalty;
    double presence_penalty;
    double dry_multiplier;
    double dry_base;
} ms_chat_generation_params;

enum {
    MS_CHAT_FINISH_REASON_STOP = 0,
    MS_CHAT_FINISH_REASON_LENGTH = 1,
    MS_CHAT_FINISH_REASON_EOS = 2
};

typedef struct ms_chat_generation_result {
    ms_baseline_benchmark benchmark;
    uint32_t prompt_tokens;
    uint32_t completion_tokens;
    uint32_t finish_reason;
    uint32_t actual_seed;
} ms_chat_generation_result;

typedef struct ms_runtime_chat_session ms_runtime_chat_session;

typedef void (*ms_runtime_log_callback)(const char * message, void * user_data);
typedef int32_t (*ms_chat_stream_callback)(
    const char * text_delta,
    const char * reasoning_delta,
    void * user_data);

typedef struct ms_runtime_chat_session_counters {
    uint64_t model_load_count;
    uint64_t context_reset_count;
    uint64_t completion_count;
    uint64_t copied_tensor_count;
    uint64_t converted_tensor_count;
    uint64_t converted_bytes_before;
    uint64_t converted_bytes_after;
    uint64_t requested_target_count;
    uint64_t verified_target_count;
} ms_runtime_chat_session_counters;

typedef struct ms_recipe_analysis {
    uint64_t tensor_count;
    uint64_t changed_count;
    uint64_t unsupported_count;
    uint64_t missing_count;
    uint64_t unknown_quant_count;
    uint64_t current_size_bytes;
    uint64_t estimated_target_size_bytes;
} ms_recipe_analysis;

typedef struct ms_recipe_eval_result {
    double baseline_load_ms;
    double baseline_prompt_eval_ms;
    double baseline_generation_ms;
    double baseline_prompt_eval_tps;
    double baseline_token_gen_tps;
    double baseline_ttft_ms;
    double baseline_runtime_elapsed_ms;
    double baseline_nll;
    double baseline_ppl;
    double baseline_ppl_uncertainty;
    double baseline_eval_ms;
    double baseline_vram_peak_mb;
    double baseline_vram_allocated_mb;
    double recipe_nll;
    double recipe_ppl;
    double recipe_ppl_uncertainty;
    double recipe_eval_ms;
    double recipe_vram_peak_mb;
    double recipe_vram_allocated_mb;
    double ppl_delta;
    double ppl_delta_percent;
    uint64_t eval_token_count;
    uint64_t eval_sample_count;
    uint64_t skipped_sample_count;
} ms_recipe_eval_result;

typedef struct ms_standard_eval_sample {
    const char * task;
    const char * prompt;
    const char * const * choices;
    const uint64_t * choice_lengths;
    uint64_t choice_count;
    uint32_t gold_index;
    uint32_t normalize_by_choice_length;
} ms_standard_eval_sample;

typedef struct ms_standard_eval_task_result {
    char task[64];
    uint64_t sample_count;
    uint64_t baseline_correct_count;
    uint64_t recipe_correct_count;
    uint64_t correct_to_wrong_count;
    uint64_t wrong_to_correct_count;
    uint64_t same_prediction_count;
    double baseline_accuracy;
    double recipe_accuracy;
    double accuracy_delta;
    double baseline_avg_margin;
    double recipe_avg_margin;
    double margin_delta;
    double baseline_avg_correct_nll;
    double recipe_avg_correct_nll;
} ms_standard_eval_task_result;

enum {
    MS_STANDARD_EVAL_AUDIT_MAX_CHOICES = 32
};

typedef struct ms_standard_eval_sample_audit {
    uint64_t sample_index;
    char task[64];
    uint32_t choice_count;
    uint32_t gold_index;
    uint32_t has_baseline;
    uint32_t baseline_prediction_index;
    uint32_t recipe_prediction_index;
    uint32_t baseline_correct;
    uint32_t recipe_correct;
    double choice_denominators[MS_STANDARD_EVAL_AUDIT_MAX_CHOICES];
    double baseline_choice_nlls[MS_STANDARD_EVAL_AUDIT_MAX_CHOICES];
    double baseline_choice_scores[MS_STANDARD_EVAL_AUDIT_MAX_CHOICES];
    double recipe_choice_nlls[MS_STANDARD_EVAL_AUDIT_MAX_CHOICES];
    double recipe_choice_scores[MS_STANDARD_EVAL_AUDIT_MAX_CHOICES];
} ms_standard_eval_sample_audit;

MS_RUNTIME_API const char * ms_runtime_version(void);
MS_RUNTIME_API const char * ms_runtime_llama_system_info(void);
MS_RUNTIME_API const char * ms_runtime_last_error(void);
MS_RUNTIME_API void ms_runtime_reset_recipe_test_cancel(void);
MS_RUNTIME_API void ms_runtime_cancel_recipe_test(void);
MS_RUNTIME_API int32_t ms_runtime_inspect_gguf(const char * path, ms_gguf_summary * out_summary);
MS_RUNTIME_API int32_t ms_runtime_preview_tensor_values(
    const char * path,
    const char * tensor_name,
    uint64_t row_offset,
    uint64_t col_offset,
    uint64_t row_count,
    uint64_t col_count,
    float * out_values,
    uint64_t value_capacity,
    uint64_t * out_rows,
    uint64_t * out_cols,
    uint64_t * out_total_rows,
    uint64_t * out_total_cols);
MS_RUNTIME_API int32_t ms_runtime_analyze_recipe(
    const char * path,
    const ms_recipe_tensor_target * targets,
    uint64_t target_count,
    ms_recipe_analysis * out_analysis);
MS_RUNTIME_API int32_t ms_runtime_benchmark_baseline(
    const char * path,
    const char * prompt,
    uint32_t max_tokens,
    ms_baseline_benchmark * out_benchmark);
MS_RUNTIME_API int32_t ms_runtime_benchmark_user_copy(
    const char * path,
    const char * prompt,
    uint32_t max_tokens,
    ms_baseline_benchmark * out_benchmark);
MS_RUNTIME_API int32_t ms_runtime_benchmark_recipe(
    const char * path,
    const ms_recipe_tensor_target * targets,
    uint64_t target_count,
    const char * prompt,
    uint32_t max_tokens,
    ms_baseline_benchmark * out_benchmark);
MS_RUNTIME_API int32_t ms_runtime_generate_recipe(
    const char * path,
    const ms_recipe_tensor_target * targets,
    uint64_t target_count,
    const char * prompt,
    uint32_t max_tokens,
    char * out_text,
    uint64_t out_text_capacity,
    ms_baseline_benchmark * out_benchmark);
MS_RUNTIME_API int32_t ms_runtime_generate_recipe_chat(
    const char * path,
    const ms_recipe_tensor_target * targets,
    uint64_t target_count,
    const ms_chat_message * messages,
    uint64_t message_count,
    uint32_t max_tokens,
    char * out_text,
    uint64_t out_text_capacity,
    ms_baseline_benchmark * out_benchmark);
MS_RUNTIME_API int32_t ms_runtime_open_recipe_chat_session(
    const char * path,
    const ms_recipe_tensor_target * targets,
    uint64_t target_count,
    uint32_t context_tokens,
    ms_runtime_chat_session ** out_session);
MS_RUNTIME_API int32_t ms_runtime_open_recipe_chat_session_with_progress(
    const char * path,
    const ms_recipe_tensor_target * targets,
    uint64_t target_count,
    uint32_t context_tokens,
    ms_runtime_log_callback log_callback,
    void * log_user_data,
    ms_runtime_chat_session ** out_session);
MS_RUNTIME_API int32_t ms_runtime_open_recipe_chat_session_with_projector_and_progress(
    const char * path,
    const char * projector_path,
    const ms_recipe_tensor_target * targets,
    uint64_t target_count,
    uint32_t context_tokens,
    ms_runtime_log_callback log_callback,
    void * log_user_data,
    ms_runtime_chat_session ** out_session);
MS_RUNTIME_API void ms_runtime_close_recipe_chat_session(ms_runtime_chat_session * session);
MS_RUNTIME_API int32_t ms_runtime_generate_recipe_chat_session(
    ms_runtime_chat_session * session,
    const ms_chat_message * messages,
    uint64_t message_count,
    const ms_chat_generation_params * params,
    const char * const * stop_strings,
    uint64_t stop_count,
    const char * chat_template_kwargs_json,
    const char * reasoning_format,
    char * out_text,
    uint64_t out_text_capacity,
    char * out_reasoning_text,
    uint64_t out_reasoning_text_capacity,
    ms_chat_generation_result * out_result);
MS_RUNTIME_API int32_t ms_runtime_generate_recipe_chat_session_stream(
    ms_runtime_chat_session * session,
    const ms_chat_message * messages,
    uint64_t message_count,
    const ms_chat_generation_params * params,
    const char * const * stop_strings,
    uint64_t stop_count,
    const char * chat_template_kwargs_json,
    const char * reasoning_format,
    ms_chat_stream_callback stream_callback,
    void * stream_user_data,
    ms_chat_generation_result * out_result);
MS_RUNTIME_API int32_t ms_runtime_generate_recipe_chat_session_multimodal_stream(
    ms_runtime_chat_session * session,
    const ms_chat_message * messages,
    uint64_t message_count,
    const char * const * image_data_urls,
    uint64_t image_count,
    const ms_chat_generation_params * params,
    const char * const * stop_strings,
    uint64_t stop_count,
    const char * chat_template_kwargs_json,
    const char * reasoning_format,
    ms_chat_stream_callback stream_callback,
    void * stream_user_data,
    ms_chat_generation_result * out_result);
MS_RUNTIME_API int32_t ms_runtime_get_recipe_chat_session_counters(
    const ms_runtime_chat_session * session,
    ms_runtime_chat_session_counters * out_counters);
MS_RUNTIME_API int32_t ms_runtime_eval_recipe(
    const char * path,
    const ms_recipe_tensor_target * targets,
    uint64_t target_count,
    const char * const * eval_texts,
    uint64_t eval_text_count,
    uint32_t max_eval_tokens,
    const char * prompt,
    uint32_t max_tokens,
    ms_baseline_benchmark * out_benchmark,
    ms_recipe_eval_result * out_eval);
MS_RUNTIME_API int32_t ms_runtime_eval_recipe_single(
    const char * path,
    const ms_recipe_tensor_target * targets,
    uint64_t target_count,
    const char * const * eval_texts,
    uint64_t eval_text_count,
    uint32_t max_eval_tokens,
    const char * prompt,
    uint32_t max_tokens,
    ms_baseline_benchmark * out_benchmark,
    ms_recipe_eval_result * out_eval);
MS_RUNTIME_API int32_t ms_runtime_eval_recipe_standard(
    const char * path,
    const ms_recipe_tensor_target * targets,
    uint64_t target_count,
    const char * const * eval_texts,
    uint64_t eval_text_count,
    const ms_standard_eval_sample * standard_samples,
    uint64_t standard_sample_count,
    uint32_t max_eval_tokens,
    const char * prompt,
    uint32_t max_tokens,
    ms_baseline_benchmark * out_benchmark,
    ms_recipe_eval_result * out_eval,
    ms_standard_eval_task_result * out_task_results,
    uint64_t task_result_capacity,
    uint64_t * out_task_result_count,
    ms_standard_eval_sample_audit * out_sample_audits,
    uint64_t sample_audit_capacity,
    uint64_t * out_sample_audit_count);
MS_RUNTIME_API int32_t ms_runtime_eval_recipe_standard_single(
    const char * path,
    const ms_recipe_tensor_target * targets,
    uint64_t target_count,
    const char * const * eval_texts,
    uint64_t eval_text_count,
    const ms_standard_eval_sample * standard_samples,
    uint64_t standard_sample_count,
    uint32_t max_eval_tokens,
    const char * prompt,
    uint32_t max_tokens,
    ms_baseline_benchmark * out_benchmark,
    ms_recipe_eval_result * out_eval,
    ms_standard_eval_task_result * out_task_results,
    uint64_t task_result_capacity,
    uint64_t * out_task_result_count,
    ms_standard_eval_sample_audit * out_sample_audits,
    uint64_t sample_audit_capacity,
    uint64_t * out_sample_audit_count);

#ifdef __cplusplus
}
#endif
