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
} ms_baseline_benchmark;

typedef struct ms_recipe_tensor_target {
    const char * name;
    const char * target_quant;
} ms_recipe_tensor_target;

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
    double baseline_eval_ms;
    double baseline_vram_peak_mb;
    double baseline_vram_allocated_mb;
    double recipe_nll;
    double recipe_ppl;
    double recipe_eval_ms;
    double recipe_vram_peak_mb;
    double recipe_vram_allocated_mb;
    double ppl_delta;
    double ppl_delta_percent;
    uint64_t eval_token_count;
    uint64_t eval_sample_count;
    uint64_t skipped_sample_count;
} ms_recipe_eval_result;

typedef enum ms_eval_sample_type {
    MS_EVAL_SAMPLE_MULTIPLE_CHOICE = 0,
    MS_EVAL_SAMPLE_EXACT_MATCH = 1,
} ms_eval_sample_type;

typedef struct ms_eval_sample {
    const char * task_name;
    uint32_t sample_type;
    const char * prompt;
    const char * const * choices;
    uint64_t choice_count;
    uint32_t answer_index;
    const char * const * answers;
    uint64_t answer_count;
    uint32_t max_tokens;
} ms_eval_sample;

typedef struct ms_eval_sample_result {
    uint32_t correct;
    uint32_t predicted_index;
    double score;
    double elapsed_ms;
} ms_eval_sample_result;

MS_RUNTIME_API const char * ms_runtime_version(void);
MS_RUNTIME_API const char * ms_runtime_llama_system_info(void);
MS_RUNTIME_API const char * ms_runtime_last_error(void);
MS_RUNTIME_API int32_t ms_runtime_inspect_gguf(const char * path, ms_gguf_summary * out_summary);
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
MS_RUNTIME_API int32_t ms_runtime_eval_task_suite(
    const char * path,
    const ms_recipe_tensor_target * targets,
    uint64_t target_count,
    const char * const * eval_texts,
    uint64_t eval_text_count,
    uint32_t max_eval_tokens,
    const ms_eval_sample * samples,
    uint64_t sample_count,
    const char * prompt,
    uint32_t max_tokens,
    ms_baseline_benchmark * out_benchmark,
    ms_recipe_eval_result * out_eval,
    ms_eval_sample_result * out_recipe_results);
MS_RUNTIME_API int32_t ms_runtime_eval_task_suite_compare(
    const char * path,
    const ms_recipe_tensor_target * targets,
    uint64_t target_count,
    const char * const * eval_texts,
    uint64_t eval_text_count,
    uint32_t max_eval_tokens,
    const ms_eval_sample * samples,
    uint64_t sample_count,
    const char * prompt,
    uint32_t max_tokens,
    ms_baseline_benchmark * out_benchmark,
    ms_recipe_eval_result * out_eval,
    ms_eval_sample_result * out_baseline_results,
    ms_eval_sample_result * out_recipe_results);

#ifdef __cplusplus
}
#endif
