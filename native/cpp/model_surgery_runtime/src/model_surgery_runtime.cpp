#include "model_surgery_runtime.h"

#include "gguf.h"
#include "llama.h"

#include <chrono>
#include <exception>
#include <string>
#include <string_view>
#include <vector>

namespace {

thread_local std::string last_error;

void null_log_callback(ggml_log_level, const char *, void *) {
}

void clear_error() {
    last_error.clear();
}

int32_t fail(const std::string & message) {
    last_error = message;
    return -1;
}

double elapsed_ms(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

bool ensure_backend_initialized() {
    static const bool initialized = [] {
        llama_log_set(null_log_callback, nullptr);
        llama_backend_init();
        return true;
    }();
    return initialized;
}

bool parse_quant_type(const char * value, ggml_type & out_type) {
    if (value == nullptr) {
        return false;
    }

    const std::string_view quant(value);
    if (quant == "F32") {
        out_type = GGML_TYPE_F32;
    } else if (quant == "BF16") {
        out_type = GGML_TYPE_BF16;
    } else if (quant == "F16") {
        out_type = GGML_TYPE_F16;
    } else if (quant == "Q8_0") {
        out_type = GGML_TYPE_Q8_0;
    } else if (quant == "Q6_K") {
        out_type = GGML_TYPE_Q6_K;
    } else if (quant == "Q5_K" || quant == "Q5_K_M") {
        out_type = GGML_TYPE_Q5_K;
    } else if (quant == "Q4_K" || quant == "Q4_K_M") {
        out_type = GGML_TYPE_Q4_K;
    } else if (quant == "Q3_K" || quant == "Q3_K_M") {
        out_type = GGML_TYPE_Q3_K;
    } else if (quant == "Q2_K") {
        out_type = GGML_TYPE_Q2_K;
    } else {
        return false;
    }

    return true;
}

uint64_t estimate_type_size(size_t current_size, ggml_type current_type, ggml_type target_type) {
    const double current_bpw = static_cast<double>(ggml_type_size(current_type)) / static_cast<double>(ggml_blck_size(current_type));
    const double target_bpw = static_cast<double>(ggml_type_size(target_type)) / static_cast<double>(ggml_blck_size(target_type));
    if (current_bpw <= 0.0 || target_bpw <= 0.0) {
        return static_cast<uint64_t>(current_size);
    }
    return static_cast<uint64_t>((static_cast<double>(current_size) * target_bpw / current_bpw) + 0.5);
}

} // namespace

extern "C" {

const char * ms_runtime_version(void) {
    return "model-surgery-runtime/0.1";
}

const char * ms_runtime_llama_system_info(void) {
    return llama_print_system_info();
}

const char * ms_runtime_last_error(void) {
    return last_error.c_str();
}

int32_t ms_runtime_inspect_gguf(const char * path, ms_gguf_summary * out_summary) {
    clear_error();

    if (path == nullptr || path[0] == '\0') {
        return fail("GGUF path is empty");
    }

    if (out_summary == nullptr) {
        return fail("summary output pointer is null");
    }

    try {
        gguf_init_params params = {};
        params.no_alloc = true;
        params.ctx = nullptr;

        gguf_context * ctx = gguf_init_from_file(path, params);
        if (ctx == nullptr) {
            return fail(std::string("failed to open GGUF: ") + path);
        }

        out_summary->version = gguf_get_version(ctx);
        out_summary->tensor_count = static_cast<uint64_t>(gguf_get_n_tensors(ctx));
        out_summary->metadata_count = static_cast<uint64_t>(gguf_get_n_kv(ctx));
        out_summary->alignment = static_cast<uint64_t>(gguf_get_alignment(ctx));
        out_summary->data_offset = static_cast<uint64_t>(gguf_get_data_offset(ctx));

        gguf_free(ctx);
        return 0;
    } catch (const std::exception & err) {
        return fail(err.what());
    } catch (...) {
        return fail("unknown native runtime error");
    }
}

int32_t ms_runtime_analyze_recipe(
    const char * path,
    const ms_recipe_tensor_target * targets,
    uint64_t target_count,
    ms_recipe_analysis * out_analysis) {
    clear_error();

    if (path == nullptr || path[0] == '\0') {
        return fail("GGUF path is empty");
    }

    if (targets == nullptr && target_count > 0) {
        return fail("recipe target pointer is null");
    }

    if (out_analysis == nullptr) {
        return fail("recipe analysis output pointer is null");
    }

    try {
        gguf_init_params params = {};
        params.no_alloc = true;
        params.ctx = nullptr;

        gguf_context * ctx = gguf_init_from_file(path, params);
        if (ctx == nullptr) {
            return fail(std::string("failed to open GGUF: ") + path);
        }

        ms_recipe_analysis analysis = {};
        analysis.tensor_count = target_count;

        for (int64_t i = 0; i < gguf_get_n_tensors(ctx); ++i) {
            analysis.current_size_bytes += static_cast<uint64_t>(gguf_get_tensor_size(ctx, i));
        }

        for (uint64_t i = 0; i < target_count; ++i) {
            const ms_recipe_tensor_target & target = targets[i];
            if (target.name == nullptr || target.name[0] == '\0') {
                analysis.missing_count += 1;
                continue;
            }

            ggml_type target_type = GGML_TYPE_COUNT;
            if (!parse_quant_type(target.target_quant, target_type)) {
                analysis.unknown_quant_count += 1;
                continue;
            }

            const int64_t tensor_id = gguf_find_tensor(ctx, target.name);
            if (tensor_id < 0) {
                analysis.missing_count += 1;
                continue;
            }

            const ggml_type current_type = gguf_get_tensor_type(ctx, tensor_id);
            const size_t current_size = gguf_get_tensor_size(ctx, tensor_id);
            if (current_type != target_type) {
                analysis.changed_count += 1;
                // Changed-tensor conversion is intentionally not implemented yet.
                analysis.unsupported_count += 1;
            }
            analysis.estimated_target_size_bytes += estimate_type_size(current_size, current_type, target_type);
        }

        gguf_free(ctx);
        *out_analysis = analysis;
        return 0;
    } catch (const std::exception & err) {
        return fail(err.what());
    } catch (...) {
        return fail("unknown native recipe analysis error");
    }
}

int32_t ms_runtime_benchmark_baseline(
    const char * path,
    const char * prompt,
    uint32_t max_tokens,
    ms_baseline_benchmark * out_benchmark) {
    clear_error();

    if (path == nullptr || path[0] == '\0') {
        return fail("GGUF path is empty");
    }

    if (prompt == nullptr || prompt[0] == '\0') {
        return fail("benchmark prompt is empty");
    }

    if (out_benchmark == nullptr) {
        return fail("benchmark output pointer is null");
    }

    try {
        ensure_backend_initialized();

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = -1;
        model_params.use_mmap = true;

        const auto load_start = std::chrono::steady_clock::now();
        llama_model * model = llama_model_load_from_file(path, model_params);
        const auto load_end = std::chrono::steady_clock::now();
        if (model == nullptr) {
            return fail(std::string("failed to load model: ") + path);
        }

        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int32_t prompt_len = static_cast<int32_t>(std::string(prompt).size());
        int32_t token_count = llama_tokenize(vocab, prompt, prompt_len, nullptr, 0, true, true);
        if (token_count == INT32_MIN) {
            llama_model_free(model);
            return fail("prompt tokenization overflowed");
        }
        if (token_count < 0) {
            token_count = -token_count;
        }
        if (token_count <= 0) {
            llama_model_free(model);
            return fail("prompt produced no tokens");
        }

        std::vector<llama_token> prompt_tokens(static_cast<size_t>(token_count));
        const int32_t actual_tokens = llama_tokenize(
            vocab,
            prompt,
            prompt_len,
            prompt_tokens.data(),
            token_count,
            true,
            true);
        if (actual_tokens <= 0) {
            llama_model_free(model);
            return fail("failed to tokenize prompt");
        }
        prompt_tokens.resize(static_cast<size_t>(actual_tokens));

        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = 512;
        ctx_params.n_batch = 512;
        ctx_params.n_ubatch = 512;
        ctx_params.n_threads = 0;
        ctx_params.n_threads_batch = 0;
        ctx_params.no_perf = true;

        llama_context * ctx = llama_init_from_model(model, ctx_params);
        if (ctx == nullptr) {
            llama_model_free(model);
            return fail("failed to create llama context");
        }

        const auto prompt_start = std::chrono::steady_clock::now();
        const int prompt_res = llama_decode(
            ctx,
            llama_batch_get_one(prompt_tokens.data(), static_cast<int32_t>(prompt_tokens.size())));
        if (prompt_res != 0) {
            llama_free(ctx);
            llama_model_free(model);
            return fail("failed to decode prompt");
        }
        llama_synchronize(ctx);
        const auto prompt_end = std::chrono::steady_clock::now();

        llama_sampler * sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

        uint32_t generated = 0;
        const auto generation_start = std::chrono::steady_clock::now();
        for (uint32_t i = 0; i < max_tokens; ++i) {
            llama_token token = llama_sampler_sample(sampler, ctx, -1);
            if (llama_vocab_is_eog(vocab, token)) {
                break;
            }

            llama_batch batch = llama_batch_get_one(&token, 1);
            const int gen_res = llama_decode(ctx, batch);
            if (gen_res != 0) {
                llama_sampler_free(sampler);
                llama_free(ctx);
                llama_model_free(model);
                return fail("failed to decode generated token");
            }
            llama_synchronize(ctx);
            ++generated;
        }
        const auto generation_end = std::chrono::steady_clock::now();

        llama_sampler_free(sampler);
        llama_free(ctx);
        llama_model_free(model);

        const double load_time_ms = elapsed_ms(load_start, load_end);
        const double prompt_time_ms = elapsed_ms(prompt_start, prompt_end);
        const double generation_time_ms = elapsed_ms(generation_start, generation_end);

        out_benchmark->load_ms = load_time_ms;
        out_benchmark->prompt_eval_ms = prompt_time_ms;
        out_benchmark->generation_ms = generation_time_ms;
        out_benchmark->prompt_eval_tps = prompt_time_ms > 0.0
            ? (static_cast<double>(prompt_tokens.size()) * 1000.0 / prompt_time_ms)
            : 0.0;
        out_benchmark->token_gen_tps = generation_time_ms > 0.0
            ? (static_cast<double>(generated) * 1000.0 / generation_time_ms)
            : 0.0;
        out_benchmark->ttft_ms = prompt_time_ms;
        out_benchmark->prompt_tokens = static_cast<uint32_t>(prompt_tokens.size());
        out_benchmark->generated_tokens = generated;

        return 0;
    } catch (const std::exception & err) {
        return fail(err.what());
    } catch (...) {
        return fail("unknown native baseline benchmark error");
    }
}

} // extern "C"
