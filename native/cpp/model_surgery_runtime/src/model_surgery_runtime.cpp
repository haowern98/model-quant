#include "model_surgery_runtime.h"

#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"

#include <chrono>
#include <cstdint>
#include <exception>
#include <fstream>
#include <ios>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
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

const char * display_quant_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32: return "F32";
        case GGML_TYPE_BF16: return "BF16";
        case GGML_TYPE_F16: return "F16";
        case GGML_TYPE_Q8_0: return "Q8_0";
        case GGML_TYPE_Q6_K: return "Q6_K";
        case GGML_TYPE_Q5_K: return "Q5_K";
        case GGML_TYPE_Q4_K: return "Q4_K";
        case GGML_TYPE_Q3_K: return "Q3_K";
        case GGML_TYPE_Q2_K: return "Q2_K";
        default: return ggml_type_name(type);
    }
}

uint64_t estimate_type_size(size_t current_size, ggml_type current_type, ggml_type target_type) {
    const double current_bpw = static_cast<double>(ggml_type_size(current_type)) / static_cast<double>(ggml_blck_size(current_type));
    const double target_bpw = static_cast<double>(ggml_type_size(target_type)) / static_cast<double>(ggml_blck_size(target_type));
    if (current_bpw <= 0.0 || target_bpw <= 0.0) {
        return static_cast<uint64_t>(current_size);
    }
    return static_cast<uint64_t>((static_cast<double>(current_size) * target_bpw / current_bpw) + 0.5);
}

std::string join_preview(const std::vector<std::string> & values, size_t limit) {
    std::string result;
    const size_t count = values.size() < limit ? values.size() : limit;
    for (size_t i = 0; i < count; ++i) {
        if (!result.empty()) {
            result += "; ";
        }
        result += values[i];
    }
    if (values.size() > limit) {
        result += "; ...";
    }
    return result;
}

bool build_recipe_targets(
    const ms_recipe_tensor_target * targets,
    uint64_t target_count,
    std::unordered_map<std::string, ggml_type> & out_targets) {
    out_targets.clear();
    out_targets.reserve(static_cast<size_t>(target_count));

    for (uint64_t i = 0; i < target_count; ++i) {
        const ms_recipe_tensor_target & target = targets[i];
        if (target.name == nullptr || target.name[0] == '\0') {
            fail("recipe target contains an empty tensor name");
            return false;
        }

        ggml_type target_type = GGML_TYPE_COUNT;
        if (!parse_quant_type(target.target_quant, target_type)) {
            fail(
                std::string("recipe target has unknown quant for ")
                + target.name + ": "
                + (target.target_quant == nullptr ? "(null)" : target.target_quant));
            return false;
        }

        if (!out_targets.emplace(target.name, target_type).second) {
            fail(std::string("recipe target has duplicate tensor entry: ") + target.name);
            return false;
        }
    }

    return true;
}

bool validate_recipe_for_user_copy(
    gguf_context * metadata,
    const std::unordered_map<std::string, ggml_type> & target_types) {
    std::vector<std::string> missing;
    std::vector<std::string> untargeted;
    std::vector<std::string> changed;

    for (int64_t i = 0; i < gguf_get_n_tensors(metadata); ++i) {
        const char * name = gguf_get_tensor_name(metadata, i);
        if (target_types.find(name) == target_types.end()) {
            untargeted.push_back(name);
        }
    }

    for (const auto & entry : target_types) {
        const std::string & name = entry.first;
        const ggml_type target_type = entry.second;
        const int64_t tensor_id = gguf_find_tensor(metadata, name.c_str());
        if (tensor_id < 0) {
            missing.push_back(name);
            continue;
        }

        const ggml_type current_type = gguf_get_tensor_type(metadata, tensor_id);
        if (current_type != target_type) {
            changed.push_back(
                name + " " + display_quant_type(current_type) + "->" + display_quant_type(target_type));
        }
    }

    if (!missing.empty()) {
        fail("Recipe does not match source GGUF; missing tensor target(s): " + join_preview(missing, 5));
        return false;
    }

    if (!untargeted.empty()) {
        fail("Recipe does not cover source GGUF tensor(s): " + join_preview(untargeted, 5));
        return false;
    }

    if (!changed.empty()) {
        fail(
            "Recipe contains unsupported changed tensor target(s): "
            + join_preview(changed, 5)
            + ". Phase 1 only supports unchanged tensor copies; conversion is not implemented yet.");
        return false;
    }

    return true;
}

struct UserCopyTensorReader {
    gguf_context * metadata = nullptr;
    std::ifstream file;
    std::vector<char> buffer;
    std::unordered_map<std::string, ggml_type> target_types;
    size_t data_offset = 0;
    uint64_t copied_tensors = 0;
};

void copy_user_tensor_data(ggml_tensor * tensor, void * userdata) {
    auto * reader = static_cast<UserCopyTensorReader *>(userdata);
    if (reader == nullptr || reader->metadata == nullptr) {
        throw std::runtime_error("user tensor copy reader is not initialized");
    }
    if (tensor == nullptr) {
        throw std::runtime_error("user tensor copy received an invalid tensor");
    }

    const char * name = ggml_get_name(tensor);
    const int64_t tensor_id = gguf_find_tensor(reader->metadata, name);
    if (tensor_id < 0) {
        throw std::runtime_error(std::string("source GGUF is missing tensor: ") + name);
    }

    const ggml_type current_type = gguf_get_tensor_type(reader->metadata, tensor_id);
    const auto target = reader->target_types.find(name);
    if (target != reader->target_types.end() && target->second != current_type) {
        throw std::runtime_error(
            std::string("unsupported tensor conversion for ")
            + name + ": " + display_quant_type(current_type)
            + "->" + display_quant_type(target->second));
    }

    const size_t expected_size = ggml_nbytes(tensor);
    const size_t source_size = gguf_get_tensor_size(reader->metadata, tensor_id);
    if (expected_size != source_size) {
        throw std::runtime_error(
            std::string("tensor size mismatch for ") + name
            + ": expected " + std::to_string(expected_size)
            + " bytes, source has " + std::to_string(source_size));
    }

    const size_t source_offset = reader->data_offset + gguf_get_tensor_offset(reader->metadata, tensor_id);
    reader->file.clear();
    reader->file.seekg(static_cast<std::streamoff>(source_offset), std::ios::beg);
    if (!reader->file.good()) {
        throw std::runtime_error(std::string("failed to seek tensor data for: ") + name);
    }

    constexpr size_t max_chunk_size = 64ull * 1024ull * 1024ull;
    reader->buffer.resize(expected_size < max_chunk_size ? expected_size : max_chunk_size);

    size_t copied = 0;
    while (copied < expected_size) {
        const size_t chunk_size = (expected_size - copied) < reader->buffer.size()
            ? (expected_size - copied)
            : reader->buffer.size();
        reader->file.read(reader->buffer.data(), static_cast<std::streamsize>(chunk_size));
        if (reader->file.gcount() != static_cast<std::streamsize>(chunk_size)) {
            throw std::runtime_error(std::string("failed to read tensor data for: ") + name);
        }
        ggml_backend_tensor_set(tensor, reader->buffer.data(), copied, chunk_size);
        copied += chunk_size;
    }

    reader->copied_tensors += 1;
}

int32_t run_loaded_model_benchmark(
    llama_model * model,
    const char * prompt,
    uint32_t max_tokens,
    double load_time_ms,
    ms_baseline_benchmark * out_benchmark) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t prompt_len = static_cast<int32_t>(std::string(prompt).size());
    int32_t token_count = llama_tokenize(vocab, prompt, prompt_len, nullptr, 0, true, true);
    if (token_count == INT32_MIN) {
        return fail("prompt tokenization overflowed");
    }
    if (token_count < 0) {
        token_count = -token_count;
    }
    if (token_count <= 0) {
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

    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
        llama_init_from_model(model, ctx_params),
        llama_free);
    if (ctx == nullptr) {
        return fail("failed to create llama context");
    }

    const auto prompt_start = std::chrono::steady_clock::now();
    const int prompt_res = llama_decode(
        ctx.get(),
        llama_batch_get_one(prompt_tokens.data(), static_cast<int32_t>(prompt_tokens.size())));
    if (prompt_res != 0) {
        return fail("failed to decode prompt");
    }
    llama_synchronize(ctx.get());
    const auto prompt_end = std::chrono::steady_clock::now();

    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(
        llama_sampler_chain_init(llama_sampler_chain_default_params()),
        llama_sampler_free);
    llama_sampler_chain_add(sampler.get(), llama_sampler_init_greedy());

    uint32_t generated = 0;
    const auto generation_start = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < max_tokens; ++i) {
        llama_token token = llama_sampler_sample(sampler.get(), ctx.get(), -1);
        if (llama_vocab_is_eog(vocab, token)) {
            break;
        }

        llama_batch batch = llama_batch_get_one(&token, 1);
        const int gen_res = llama_decode(ctx.get(), batch);
        if (gen_res != 0) {
            return fail("failed to decode generated token");
        }
        llama_synchronize(ctx.get());
        ++generated;
    }
    const auto generation_end = std::chrono::steady_clock::now();

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

        const double load_time_ms = elapsed_ms(load_start, load_end);
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model_guard(model, llama_model_free);
        return run_loaded_model_benchmark(model_guard.get(), prompt, max_tokens, load_time_ms, out_benchmark);
    } catch (const std::exception & err) {
        return fail(err.what());
    } catch (...) {
        return fail("unknown native baseline benchmark error");
    }
}

int32_t ms_runtime_benchmark_user_copy(
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

        gguf_init_params gguf_params = {};
        gguf_params.no_alloc = true;
        gguf_params.ctx = nullptr;

        std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(
            gguf_init_from_file(path, gguf_params),
            gguf_free);
        if (metadata == nullptr) {
            return fail(std::string("failed to open GGUF metadata: ") + path);
        }

        UserCopyTensorReader reader = {};
        reader.metadata = metadata.get();
        reader.data_offset = gguf_get_data_offset(metadata.get());
        reader.file.open(path, std::ios::binary);
        if (!reader.file.is_open()) {
            return fail(std::string("failed to open GGUF tensor data: ") + path);
        }

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = -1;
        model_params.use_mmap = false;

        const auto load_start = std::chrono::steady_clock::now();
        llama_model * model = llama_model_init_from_user(
            metadata.get(),
            copy_user_tensor_data,
            &reader,
            model_params);
        const auto load_end = std::chrono::steady_clock::now();
        if (model == nullptr) {
            return fail(std::string("failed to load user-copy model: ") + path);
        }

        const double load_time_ms = elapsed_ms(load_start, load_end);
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model_guard(model, llama_model_free);
        return run_loaded_model_benchmark(model_guard.get(), prompt, max_tokens, load_time_ms, out_benchmark);
    } catch (const std::exception & err) {
        return fail(err.what());
    } catch (...) {
        return fail("unknown native user-copy benchmark error");
    }
}

int32_t ms_runtime_benchmark_recipe(
    const char * path,
    const ms_recipe_tensor_target * targets,
    uint64_t target_count,
    const char * prompt,
    uint32_t max_tokens,
    ms_baseline_benchmark * out_benchmark) {
    clear_error();

    if (path == nullptr || path[0] == '\0') {
        return fail("GGUF path is empty");
    }

    if (targets == nullptr && target_count > 0) {
        return fail("recipe target pointer is null");
    }

    if (prompt == nullptr || prompt[0] == '\0') {
        return fail("benchmark prompt is empty");
    }

    if (out_benchmark == nullptr) {
        return fail("benchmark output pointer is null");
    }

    try {
        ensure_backend_initialized();

        std::unordered_map<std::string, ggml_type> target_types;
        if (!build_recipe_targets(targets, target_count, target_types)) {
            return -1;
        }

        gguf_init_params gguf_params = {};
        gguf_params.no_alloc = true;
        gguf_params.ctx = nullptr;

        std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(
            gguf_init_from_file(path, gguf_params),
            gguf_free);
        if (metadata == nullptr) {
            return fail(std::string("failed to open GGUF metadata: ") + path);
        }

        if (!validate_recipe_for_user_copy(metadata.get(), target_types)) {
            return -1;
        }

        UserCopyTensorReader reader = {};
        reader.metadata = metadata.get();
        reader.target_types = std::move(target_types);
        reader.data_offset = gguf_get_data_offset(metadata.get());
        reader.file.open(path, std::ios::binary);
        if (!reader.file.is_open()) {
            return fail(std::string("failed to open GGUF tensor data: ") + path);
        }

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = -1;
        model_params.use_mmap = false;

        const auto load_start = std::chrono::steady_clock::now();
        llama_model * model = llama_model_init_from_user(
            metadata.get(),
            copy_user_tensor_data,
            &reader,
            model_params);
        const auto load_end = std::chrono::steady_clock::now();
        if (model == nullptr) {
            return fail(std::string("failed to load recipe model: ") + path);
        }

        const double load_time_ms = elapsed_ms(load_start, load_end);
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model_guard(model, llama_model_free);
        return run_loaded_model_benchmark(model_guard.get(), prompt, max_tokens, load_time_ms, out_benchmark);
    } catch (const std::exception & err) {
        return fail(err.what());
    } catch (...) {
        return fail("unknown native recipe benchmark error");
    }
}

} // extern "C"
