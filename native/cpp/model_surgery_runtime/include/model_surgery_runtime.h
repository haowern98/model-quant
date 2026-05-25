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
    uint32_t prompt_tokens;
    uint32_t generated_tokens;
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

#ifdef __cplusplus
}
#endif
