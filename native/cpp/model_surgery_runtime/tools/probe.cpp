#include "model_surgery_runtime.h"

#include <iostream>

int main(int argc, char ** argv) {
    std::cout << ms_runtime_version() << "\n";
    std::cout << ms_runtime_llama_system_info() << "\n";
    std::cout.flush();

    if (argc < 2) {
        return 0;
    }

    ms_gguf_summary summary = {};
    const int32_t result = ms_runtime_inspect_gguf(argv[1], &summary);
    if (result != 0) {
        std::cerr << ms_runtime_last_error() << "\n";
        return 1;
    }

    std::cout
        << "gguf_version=" << summary.version
        << " tensors=" << summary.tensor_count
        << " metadata=" << summary.metadata_count
        << " alignment=" << summary.alignment
        << " data_offset=" << summary.data_offset
        << "\n";
    std::cout.flush();

    ms_recipe_analysis analysis = {};
    const int32_t analysis_result = ms_runtime_analyze_recipe(argv[1], nullptr, 0, &analysis);
    if (analysis_result != 0) {
        std::cerr << ms_runtime_last_error() << "\n";
        return 1;
    }
    std::cout
        << "analysis_current_size_mb=" << (analysis.current_size_bytes / 1024.0 / 1024.0)
        << " changed=" << analysis.changed_count
        << " unsupported=" << analysis.unsupported_count
        << "\n";
    std::cout.flush();

    ms_baseline_benchmark benchmark = {};
    const int32_t bench_result = ms_runtime_benchmark_baseline(
        argv[1],
        "The capital of France is",
        8,
        &benchmark);
    if (bench_result != 0) {
        std::cerr << ms_runtime_last_error() << "\n";
        return 1;
    }

    std::cout
        << "load_ms=" << benchmark.load_ms
        << " prompt_tps=" << benchmark.prompt_eval_tps
        << " gen_tps=" << benchmark.token_gen_tps
        << " prompt_tokens=" << benchmark.prompt_tokens
        << " generated_tokens=" << benchmark.generated_tokens
        << "\n";
    std::cout.flush();

    ms_baseline_benchmark user_copy_benchmark = {};
    const int32_t user_copy_result = ms_runtime_benchmark_user_copy(
        argv[1],
        "The capital of France is",
        8,
        &user_copy_benchmark);
    if (user_copy_result != 0) {
        std::cerr << ms_runtime_last_error() << "\n";
        return 1;
    }

    std::cout
        << "user_copy_load_ms=" << user_copy_benchmark.load_ms
        << " user_copy_prompt_tps=" << user_copy_benchmark.prompt_eval_tps
        << " user_copy_gen_tps=" << user_copy_benchmark.token_gen_tps
        << " user_copy_prompt_tokens=" << user_copy_benchmark.prompt_tokens
        << " user_copy_generated_tokens=" << user_copy_benchmark.generated_tokens
        << "\n";
    std::cout.flush();

    return 0;
}
