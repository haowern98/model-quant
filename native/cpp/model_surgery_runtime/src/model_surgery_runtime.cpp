#include "model_surgery_runtime.h"

#include "chat.h"
#include "base64.hpp"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"
#include "sampling.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <exception>
#include <fstream>
#include <ios>
#include <limits>
#include <memory>
#include <map>
#include <string>
#include <string_view>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#if defined(MODEL_SURGERY_RUNTIME_CUDA_PROFILING)
#include <cuda_runtime.h>
#endif

namespace {

thread_local std::string last_error;
std::atomic<bool> recipe_test_cancel_flag{false};

bool recipe_test_cancel_requested() {
    return recipe_test_cancel_flag.load(std::memory_order_relaxed);
}

void throw_if_recipe_test_cancelled() {
    if (recipe_test_cancel_requested()) {
        throw std::runtime_error("Recipe test cancelled");
    }
}

bool recipe_test_load_progress(float, void *) {
    return !recipe_test_cancel_requested();
}

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

double benchmark_runtime_elapsed_ms(const ms_baseline_benchmark & benchmark) {
    return benchmark.load_ms + benchmark.prompt_eval_ms + benchmark.generation_ms;
}

bool read_cuda_used_mb(double & out_used_mb) {
#if defined(MODEL_SURGERY_RUNTIME_CUDA_PROFILING)
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    const cudaError_t result = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (result != cudaSuccess) {
        return false;
    }

    out_used_mb = static_cast<double>(total_bytes - free_bytes) / (1024.0 * 1024.0);
    return true;
#else
    out_used_mb = 0.0;
    return false;
#endif
}

struct VramTracker {
    bool available = false;
    double baseline_mb = 0.0;
    double current_mb = 0.0;
    double peak_mb = 0.0;

    void reset() {
        double used_mb = 0.0;
        available = read_cuda_used_mb(used_mb);
        baseline_mb = available ? used_mb : 0.0;
        current_mb = 0.0;
        peak_mb = 0.0;
    }

    void sample() {
        if (!available) {
            return;
        }

        double used_mb = 0.0;
        if (!read_cuda_used_mb(used_mb)) {
            available = false;
            current_mb = 0.0;
            peak_mb = 0.0;
            return;
        }

        current_mb = used_mb > baseline_mb ? used_mb - baseline_mb : 0.0;
        if (current_mb > peak_mb) {
            peak_mb = current_mb;
        }
    }
};

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
    } else if (quant == "Q5_1") {
        out_type = GGML_TYPE_Q5_1;
    } else if (quant == "Q5_0") {
        out_type = GGML_TYPE_Q5_0;
    } else if (quant == "Q4_K" || quant == "Q4_K_M") {
        out_type = GGML_TYPE_Q4_K;
    } else if (quant == "Q4_1") {
        out_type = GGML_TYPE_Q4_1;
    } else if (quant == "Q4_0") {
        out_type = GGML_TYPE_Q4_0;
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
        case GGML_TYPE_Q5_1: return "Q5_1";
        case GGML_TYPE_Q5_0: return "Q5_0";
        case GGML_TYPE_Q4_K: return "Q4_K";
        case GGML_TYPE_Q4_1: return "Q4_1";
        case GGML_TYPE_Q4_0: return "Q4_0";
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

bool is_runtime_quantizable_tensor(std::string_view name) {
    return name.find("bias") == std::string_view::npos
        && name.find("norm") == std::string_view::npos
        && name.find("rope") == std::string_view::npos
        && name.find("scale") == std::string_view::npos;
}

bool is_supported_recipe_target(ggml_type target_type) {
    switch (target_type) {
        case GGML_TYPE_BF16:
        case GGML_TYPE_F16:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q2_K:
            return true;
        default:
            return false;
    }
}

enum class RecipeQuantFamily {
    Full,
    Q8,
    Legacy,
    K,
    Other,
};

RecipeQuantFamily recipe_quant_family(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_BF16:
        case GGML_TYPE_F16:
            return RecipeQuantFamily::Full;
        case GGML_TYPE_Q8_0:
            return RecipeQuantFamily::Q8;
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q4_0:
            return RecipeQuantFamily::Legacy;
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q2_K:
            return RecipeQuantFamily::K;
        default:
            return RecipeQuantFamily::Other;
    }
}

bool recipe_quant_family_allows(ggml_type current_type, ggml_type target_type) {
    const RecipeQuantFamily current_family = recipe_quant_family(current_type);
    const RecipeQuantFamily target_family = recipe_quant_family(target_type);

    switch (current_family) {
        case RecipeQuantFamily::Full:
            return target_family != RecipeQuantFamily::Other;
        case RecipeQuantFamily::Q8:
            return target_family == RecipeQuantFamily::Q8
                || target_family == RecipeQuantFamily::Legacy
                || target_family == RecipeQuantFamily::K;
        case RecipeQuantFamily::Legacy:
            return target_family == RecipeQuantFamily::Legacy;
        case RecipeQuantFamily::K:
            return target_family == RecipeQuantFamily::K;
        case RecipeQuantFamily::Other:
            return false;
    }

    return false;
}

bool can_decode_source_to_f32(ggml_type current_type) {
    if (current_type == GGML_TYPE_F32) {
        return true;
    }

    const struct ggml_type_traits * traits = ggml_get_type_traits(current_type);
    return traits != nullptr && traits->to_float != nullptr;
}

bool supports_recipe_conversion(std::string_view name, ggml_type current_type, ggml_type target_type) {
    if (!is_supported_recipe_target(target_type)) {
        return false;
    }

    if (estimate_type_size(1024, current_type, target_type) > 1024) {
        return false;
    }

    if (!recipe_quant_family_allows(current_type, target_type)) {
        return false;
    }

    if (!is_runtime_quantizable_tensor(name)) {
        return false;
    }

    return can_decode_source_to_f32(current_type);
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

bool validate_recipe_for_user_model(
    gguf_context * source_metadata,
    const std::unordered_map<std::string, ggml_type> & target_types) {
    std::vector<std::string> missing;
    std::vector<std::string> untargeted;
    std::vector<std::string> unsupported;

    for (int64_t i = 0; i < gguf_get_n_tensors(source_metadata); ++i) {
        const char * name = gguf_get_tensor_name(source_metadata, i);
        if (target_types.find(name) == target_types.end()) {
            untargeted.push_back(name);
        }
    }

    for (const auto & entry : target_types) {
        const std::string & name = entry.first;
        const ggml_type target_type = entry.second;
        const int64_t tensor_id = gguf_find_tensor(source_metadata, name.c_str());
        if (tensor_id < 0) {
            missing.push_back(name);
            continue;
        }

        const ggml_type current_type = gguf_get_tensor_type(source_metadata, tensor_id);
        if (current_type != target_type && !supports_recipe_conversion(name, current_type, target_type)) {
            unsupported.push_back(
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

    if (!unsupported.empty()) {
        fail(
            "Recipe contains unsupported changed tensor target(s): "
            + join_preview(unsupported, 5)
            + ". Test Recipe supports equal-or-smaller F16/BF16/Q8_0/K-quant targets for compatible weight tensors.");
        return false;
    }

    return true;
}

void apply_recipe_tensor_types(
    gguf_context * model_metadata,
    const std::unordered_map<std::string, ggml_type> & target_types) {
    for (const auto & entry : target_types) {
        gguf_set_tensor_type(model_metadata, entry.first.c_str(), entry.second);
    }
}

struct UserCopyTensorReader {
    gguf_context * source_metadata = nullptr;
    std::ifstream file;
    std::vector<char> buffer;
    std::vector<char> source_row_buffer;
    std::vector<float> f32_row_buffer;
    std::vector<char> quantized_row_buffer;
    std::unordered_map<std::string, ggml_type> target_types;
    std::unordered_map<std::string, ggml_type> loaded_target_types;
    size_t data_offset = 0;
    uint64_t copied_tensors = 0;
    uint64_t converted_tensors = 0;
    uint64_t converted_bytes_before = 0;
    uint64_t converted_bytes_after = 0;
};

struct RuntimeLogSink {
    ms_runtime_log_callback callback = nullptr;
    void * user_data = nullptr;

    void emit(const std::string & message) const {
        if (callback != nullptr) {
            callback(message.c_str(), user_data);
        }
    }
};

struct ParsedChatOutput {
    std::string visible_text;
    std::string reasoning_text;
};

struct RecipeTargetVerification {
    uint64_t requested_count = 0;
    uint64_t verified_count = 0;
    uint64_t mismatch_count = 0;
    std::string first_mismatch;
};

struct PerplexityScore {
    double total_nll = 0.0;
    double ppl = 0.0;
    double ppl_uncertainty = 0.0;
    double eval_ms = 0.0;
    double vram_peak_mb = 0.0;
    double vram_allocated_mb = 0.0;
    uint64_t token_count = 0;
    uint64_t sample_count = 0;
    uint64_t skipped_count = 0;
};

constexpr uint32_t LLAMA_PPL_CONTEXT_TOKENS = 512;
constexpr uint32_t LLAMA_GENERATION_CONTEXT_TOKENS = 8192;

struct StandardEvalSampleScore {
    std::string task;
    uint64_t sample_index = 0;
    uint32_t gold_index = 0;
    uint32_t prediction_index = 0;
    bool correct = false;
    double margin = 0.0;
    double correct_nll = 0.0;
    std::vector<double> choice_denominators;
    std::vector<double> choice_nlls;
    std::vector<double> choice_scores;
};

struct StandardEvalAccumulator {
    std::string task;
    uint64_t sample_count = 0;
    uint64_t baseline_correct_count = 0;
    uint64_t recipe_correct_count = 0;
    uint64_t correct_to_wrong_count = 0;
    uint64_t wrong_to_correct_count = 0;
    uint64_t same_prediction_count = 0;
    double baseline_margin_sum = 0.0;
    double recipe_margin_sum = 0.0;
    double baseline_correct_nll_sum = 0.0;
    double recipe_correct_nll_sum = 0.0;
};

void copy_user_tensor_data(ggml_tensor * tensor, void * userdata);

struct ModelSession {
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model;
    common_chat_templates_ptr chat_templates;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx;
    std::unique_ptr<mtmd_context, decltype(&mtmd_free)> mctx;
    VramTracker vram;
    double load_ms = 0.0;
    uint64_t copied_tensors = 0;
    uint64_t converted_tensors = 0;
    uint64_t converted_bytes_before = 0;
    uint64_t converted_bytes_after = 0;
    uint64_t requested_target_count = 0;
    uint64_t verified_target_count = 0;
    uint64_t context_reset_count = 0;

    ModelSession(llama_model * loaded_model, double load_time_ms, const VramTracker & vram_tracker)
        : model(loaded_model, llama_model_free),
          chat_templates(common_chat_templates_init(loaded_model, "")),
          ctx(nullptr, llama_free),
          mctx(nullptr, mtmd_free),
          vram(vram_tracker),
          load_ms(load_time_ms) {
    }
};

uint32_t session_context_tokens(uint32_t max_eval_tokens) {
    const uint32_t eval_limit = std::max<uint32_t>(2, max_eval_tokens == 0 ? 128 : max_eval_tokens);
    return std::max<uint32_t>(512, eval_limit + 8);
}

uint32_t session_context_tokens_for_generation(uint32_t max_tokens) {
    const uint32_t generation_limit = std::max<uint32_t>(2, max_tokens == 0 ? 128 : max_tokens);
    return std::max<uint32_t>(LLAMA_GENERATION_CONTEXT_TOKENS, generation_limit * 2);
}

uint32_t context_generation_room(uint32_t context_tokens, size_t prompt_tokens) {
    constexpr uint32_t reserved_tokens = 1;
    if (prompt_tokens + reserved_tokens >= context_tokens) {
        return 0;
    }
    return static_cast<uint32_t>(context_tokens - prompt_tokens - reserved_tokens);
}

void open_session_context(
    ModelSession & session,
    uint32_t context_tokens,
    const RuntimeLogSink * log = nullptr) {
    throw_if_recipe_test_cancelled();
    if (log != nullptr) {
        log->emit("Native runtime: creating chat context");
    }
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = context_tokens;
    ctx_params.n_batch = std::min<uint32_t>(512, context_tokens);
    ctx_params.n_ubatch = std::min<uint32_t>(512, context_tokens);
    ctx_params.n_threads = 0;
    ctx_params.n_threads_batch = 0;
    ctx_params.no_perf = true;

    session.ctx.reset(llama_init_from_model(session.model.get(), ctx_params));
    if (session.ctx == nullptr) {
        throw std::runtime_error("failed to create llama context for model session");
    }
    throw_if_recipe_test_cancelled();
    session.vram.sample();
    if (log != nullptr) {
        log->emit("Native runtime: chat context ready");
    }
}

void reset_session_context(ModelSession & session) {
    if (session.ctx == nullptr) {
        throw std::runtime_error("model session context is not initialized");
    }
    llama_memory_clear(llama_get_memory(session.ctx.get()), true);
    session.context_reset_count += 1;
}

std::unique_ptr<ModelSession> open_baseline_session(const char * path, uint32_t context_tokens) {
    throw_if_recipe_test_cancelled();
    VramTracker vram_tracker = {};
    vram_tracker.reset();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = -1;
    model_params.use_mmap = true;
    model_params.progress_callback = recipe_test_load_progress;

    const auto load_start = std::chrono::steady_clock::now();
    llama_model * model = llama_model_load_from_file(path, model_params);
    const auto load_end = std::chrono::steady_clock::now();
    if (model == nullptr) {
        throw_if_recipe_test_cancelled();
        throw std::runtime_error(std::string("failed to load model: ") + path);
    }
    throw_if_recipe_test_cancelled();
    vram_tracker.sample();

    auto session = std::make_unique<ModelSession>(model, elapsed_ms(load_start, load_end), vram_tracker);
    open_session_context(*session, context_tokens);
    return session;
}

std::unique_ptr<ModelSession> open_user_copy_session(const char * path, uint32_t context_tokens) {
    VramTracker vram_tracker = {};
    vram_tracker.reset();

    gguf_init_params gguf_params = {};
    gguf_params.no_alloc = true;
    gguf_params.ctx = nullptr;

    std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(
        gguf_init_from_file(path, gguf_params),
        gguf_free);
    if (metadata == nullptr) {
        throw std::runtime_error(std::string("failed to open GGUF metadata: ") + path);
    }

    UserCopyTensorReader reader = {};
    reader.source_metadata = metadata.get();
    reader.data_offset = gguf_get_data_offset(metadata.get());
    reader.file.open(path, std::ios::binary);
    if (!reader.file.is_open()) {
        throw std::runtime_error(std::string("failed to open GGUF tensor data: ") + path);
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
        throw std::runtime_error(std::string("failed to load user-copy model: ") + path);
    }
    vram_tracker.sample();

    auto session = std::make_unique<ModelSession>(model, elapsed_ms(load_start, load_end), vram_tracker);
    session->copied_tensors = reader.copied_tensors;
    session->converted_tensors = reader.converted_tensors;
    session->converted_bytes_before = reader.converted_bytes_before;
    session->converted_bytes_after = reader.converted_bytes_after;
    open_session_context(*session, context_tokens);
    return session;
}

std::unordered_map<std::string, ggml_type> build_source_types_for_targets(
    gguf_context * source_metadata,
    const std::unordered_map<std::string, ggml_type> & target_types) {
    std::unordered_map<std::string, ggml_type> source_types;
    for (const auto & entry : target_types) {
        const int64_t tensor_id = gguf_find_tensor(source_metadata, entry.first.c_str());
        if (tensor_id < 0) {
            continue;
        }
        source_types.emplace(entry.first, gguf_get_tensor_type(source_metadata, tensor_id));
    }
    return source_types;
}

RecipeTargetVerification verify_recipe_tensor_target_types(
    const std::unordered_map<std::string, ggml_type> & loaded_target_types,
    const std::unordered_map<std::string, ggml_type> & target_types,
    const std::unordered_map<std::string, ggml_type> & source_types) {
    RecipeTargetVerification verification = {};

    for (const auto & target : target_types) {
        const auto source = source_types.find(target.first);
        if (source == source_types.end() || source->second == target.second) {
            continue;
        }

        verification.requested_count += 1;

        const auto loaded = loaded_target_types.find(target.first);
        if (loaded == loaded_target_types.end()) {
            verification.mismatch_count += 1;
            if (verification.first_mismatch.empty()) {
                verification.first_mismatch = target.first
                    + " expected " + display_quant_type(target.second)
                    + ", loaded <missing>";
            }
            continue;
        }

        const ggml_type loaded_type = loaded->second;
        if (loaded_type == target.second) {
            verification.verified_count += 1;
            continue;
        }

        verification.mismatch_count += 1;
        if (verification.first_mismatch.empty()) {
            verification.first_mismatch = target.first
                + " expected " + display_quant_type(target.second)
                + ", loaded " + display_quant_type(loaded_type);
        }
    }

    return verification;
}

RecipeTargetVerification verify_recipe_tensor_targets_in_map(
    const std::vector<std::pair<std::string, ggml_tensor *>> & tensor_map,
    const std::unordered_map<std::string, ggml_type> & target_types,
    const std::unordered_map<std::string, ggml_type> & source_types) {
    std::unordered_map<std::string, ggml_type> loaded_target_types;
    for (const auto & entry : tensor_map) {
        if (entry.second != nullptr) {
            loaded_target_types.emplace(entry.first, entry.second->type);
        }
    }
    return verify_recipe_tensor_target_types(
        loaded_target_types,
        target_types,
        source_types);
}

std::unique_ptr<ModelSession> open_recipe_session(
    const char * path,
    std::unordered_map<std::string, ggml_type> target_types,
    uint32_t context_tokens,
    const RuntimeLogSink * log = nullptr) {
    throw_if_recipe_test_cancelled();
    gguf_init_params gguf_params = {};
    gguf_params.no_alloc = true;
    gguf_params.ctx = nullptr;

    if (log != nullptr) {
        log->emit("Native runtime: opening GGUF metadata");
    }
    std::unique_ptr<gguf_context, decltype(&gguf_free)> source_metadata(
        gguf_init_from_file(path, gguf_params),
        gguf_free);
    if (source_metadata == nullptr) {
        throw std::runtime_error(std::string("failed to open GGUF metadata: ") + path);
    }

    std::unique_ptr<gguf_context, decltype(&gguf_free)> model_metadata(
        gguf_init_from_file(path, gguf_params),
        gguf_free);
    if (model_metadata == nullptr) {
        throw std::runtime_error(std::string("failed to open GGUF model metadata: ") + path);
    }
    if (log != nullptr) {
        log->emit("Native runtime: GGUF metadata loaded");
    }

    if (!validate_recipe_for_user_model(source_metadata.get(), target_types)) {
        return nullptr;
    }
    const std::unordered_map<std::string, ggml_type> source_types = build_source_types_for_targets(
        source_metadata.get(),
        target_types);
    if (log != nullptr) {
        if (target_types.empty()) {
            log->emit("Native runtime: using source tensor types");
        } else {
            log->emit("Native runtime: applying in-memory recipe target metadata");
        }
    }
    apply_recipe_tensor_types(model_metadata.get(), target_types);

    UserCopyTensorReader reader = {};
    reader.source_metadata = source_metadata.get();
    reader.target_types = std::move(target_types);
    reader.data_offset = gguf_get_data_offset(source_metadata.get());
    if (log != nullptr) {
        log->emit("Native runtime: opening GGUF tensor data");
    }
    reader.file.open(path, std::ios::binary);
    if (!reader.file.is_open()) {
        throw std::runtime_error(std::string("failed to open GGUF tensor data: ") + path);
    }

    VramTracker vram_tracker = {};
    vram_tracker.reset();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = -1;
    model_params.use_mmap = false;
    model_params.progress_callback = recipe_test_load_progress;

    if (log != nullptr) {
        log->emit("Native runtime: loading model weights into memory");
    }
    const auto load_start = std::chrono::steady_clock::now();
    llama_model * model = llama_model_init_from_user(
        model_metadata.get(),
        copy_user_tensor_data,
        &reader,
        model_params);
    const auto load_end = std::chrono::steady_clock::now();
    if (model == nullptr) {
        throw_if_recipe_test_cancelled();
        throw std::runtime_error(std::string("failed to load recipe model: ") + path);
    }
    throw_if_recipe_test_cancelled();
    vram_tracker.sample();
    if (log != nullptr) {
        log->emit("Native runtime: model weights loaded");
    }

    const RecipeTargetVerification verification = verify_recipe_tensor_target_types(
        reader.loaded_target_types,
        reader.target_types,
        source_types);
    if (verification.mismatch_count > 0) {
        llama_model_free(model);
        throw std::runtime_error(
            "Recipe tensor verification failed: "
            + verification.first_mismatch);
    }

    auto session = std::make_unique<ModelSession>(model, elapsed_ms(load_start, load_end), vram_tracker);
    session->copied_tensors = reader.copied_tensors;
    session->converted_tensors = reader.converted_tensors;
    session->converted_bytes_before = reader.converted_bytes_before;
    session->converted_bytes_after = reader.converted_bytes_after;
    session->requested_target_count = verification.requested_count;
    session->verified_target_count = verification.verified_count;
    if (log != nullptr) {
        log->emit(
            "Native runtime: verified recipe targets "
            + std::to_string(verification.verified_count)
            + "/"
            + std::to_string(verification.requested_count));
    }
    open_session_context(*session, context_tokens, log);
    return session;
}

void read_exact_at(
    UserCopyTensorReader * reader,
    size_t offset,
    void * data,
    size_t size,
    const char * tensor_name) {
    reader->file.clear();
    reader->file.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!reader->file.good()) {
        throw std::runtime_error(std::string("failed to seek tensor data for: ") + tensor_name);
    }

    reader->file.read(static_cast<char *>(data), static_cast<std::streamsize>(size));
    if (reader->file.gcount() != static_cast<std::streamsize>(size)) {
        throw std::runtime_error(std::string("failed to read tensor data for: ") + tensor_name);
    }
}

void decode_source_row_to_f32(
    const void * source_row,
    ggml_type current_type,
    int64_t n_per_row,
    float * f32_row,
    std::string_view name) {
    if (source_row == nullptr || f32_row == nullptr) {
        throw std::runtime_error(std::string("cannot decode invalid tensor row for: ") + std::string(name));
    }

    if (current_type == GGML_TYPE_F32) {
        std::memcpy(f32_row, source_row, static_cast<size_t>(n_per_row) * sizeof(float));
        return;
    }

    const struct ggml_type_traits * traits = ggml_get_type_traits(current_type);
    if (traits == nullptr || traits->to_float == nullptr) {
        throw std::runtime_error(
            std::string("unsupported source quant for recipe conversion: ")
            + std::string(name) + " " + display_quant_type(current_type));
    }

    traits->to_float(source_row, f32_row, n_per_row);
}

void convert_source_tensor_to_quant(
    ggml_tensor * tensor,
    UserCopyTensorReader * reader,
    const char * name,
    int64_t tensor_id,
    ggml_type current_type,
    ggml_type target_type) {
    const int64_t n_per_row = tensor->ne[0];
    if (n_per_row <= 0) {
        throw std::runtime_error(std::string("cannot quantize tensor with empty row: ") + name);
    }

    const int64_t target_block = ggml_blck_size(target_type);
    if (n_per_row % target_block != 0) {
        throw std::runtime_error(
            std::string("cannot quantize tensor to ")
            + display_quant_type(target_type)
            + " because row size is not divisible by "
            + std::to_string(target_block) + ": " + name);
    }

    const int64_t element_count = ggml_nelements(tensor);
    if (element_count % n_per_row != 0) {
        throw std::runtime_error(std::string("cannot quantize tensor with irregular row layout: ") + name);
    }

    const int64_t nrows = element_count / n_per_row;
    const size_t source_row_size = ggml_row_size(current_type, n_per_row);
    const size_t target_row_size = ggml_row_size(target_type, n_per_row);
    const size_t source_size = gguf_get_tensor_size(reader->source_metadata, tensor_id);
    const size_t target_size = ggml_nbytes(tensor);

    if (source_size != source_row_size * static_cast<size_t>(nrows)) {
        throw std::runtime_error(std::string("source tensor size does not match row layout for: ") + name);
    }
    if (target_size != target_row_size * static_cast<size_t>(nrows)) {
        throw std::runtime_error(
            std::string("target tensor size does not match ")
            + display_quant_type(target_type)
            + " row layout for: " + name);
    }

    reader->source_row_buffer.resize(source_row_size);
    reader->f32_row_buffer.resize(static_cast<size_t>(n_per_row));
    reader->quantized_row_buffer.resize(target_row_size);

    const size_t source_offset = reader->data_offset + gguf_get_tensor_offset(reader->source_metadata, tensor_id);
    for (int64_t row = 0; row < nrows; ++row) {
        throw_if_recipe_test_cancelled();
        read_exact_at(
            reader,
            source_offset + static_cast<size_t>(row) * source_row_size,
            reader->source_row_buffer.data(),
            source_row_size,
            name);

        decode_source_row_to_f32(
            reader->source_row_buffer.data(),
            current_type,
            n_per_row,
            reader->f32_row_buffer.data(),
            name);

        const size_t written = ggml_quantize_chunk(
            target_type,
            reader->f32_row_buffer.data(),
            reader->quantized_row_buffer.data(),
            0,
            1,
            n_per_row,
            nullptr);
        if (written != target_row_size) {
            throw std::runtime_error(
                std::string(display_quant_type(target_type))
                + " quantized row size mismatch for: " + name);
        }

        ggml_backend_tensor_set(
            tensor,
            reader->quantized_row_buffer.data(),
            static_cast<size_t>(row) * target_row_size,
            target_row_size);
    }

    reader->converted_tensors += 1;
    reader->converted_bytes_before += source_size;
    reader->converted_bytes_after += target_size;
}

void copy_user_tensor_data(ggml_tensor * tensor, void * userdata) {
    throw_if_recipe_test_cancelled();
    auto * reader = static_cast<UserCopyTensorReader *>(userdata);
    if (reader == nullptr || reader->source_metadata == nullptr) {
        throw std::runtime_error("user tensor copy reader is not initialized");
    }
    if (tensor == nullptr) {
        throw std::runtime_error("user tensor copy received an invalid tensor");
    }

    const char * name = ggml_get_name(tensor);
    const int64_t tensor_id = gguf_find_tensor(reader->source_metadata, name);
    if (tensor_id < 0) {
        throw std::runtime_error(std::string("source GGUF is missing tensor: ") + name);
    }

    const ggml_type current_type = gguf_get_tensor_type(reader->source_metadata, tensor_id);
    const auto target = reader->target_types.find(name);
    const ggml_type target_type = target == reader->target_types.end() ? current_type : target->second;
    if (target_type != current_type) {
        reader->loaded_target_types[name] = tensor->type;
        if (supports_recipe_conversion(name, current_type, target_type)) {
            convert_source_tensor_to_quant(tensor, reader, name, tensor_id, current_type, target_type);
            return;
        }

        throw std::runtime_error(
            std::string("unsupported tensor conversion for ")
            + name + ": " + display_quant_type(current_type)
            + "->" + display_quant_type(target_type));
    }

    const size_t expected_size = ggml_nbytes(tensor);
    const size_t source_size = gguf_get_tensor_size(reader->source_metadata, tensor_id);
    if (expected_size != source_size) {
        throw std::runtime_error(
            std::string("tensor size mismatch for ") + name
            + ": expected " + std::to_string(expected_size)
            + " bytes, source has " + std::to_string(source_size));
    }

    const size_t source_offset = reader->data_offset + gguf_get_tensor_offset(reader->source_metadata, tensor_id);

    constexpr size_t max_chunk_size = 64ull * 1024ull * 1024ull;
    reader->buffer.resize(expected_size < max_chunk_size ? expected_size : max_chunk_size);

    size_t copied = 0;
    while (copied < expected_size) {
        throw_if_recipe_test_cancelled();
        const size_t chunk_size = (expected_size - copied) < reader->buffer.size()
            ? (expected_size - copied)
            : reader->buffer.size();
        read_exact_at(reader, source_offset + copied, reader->buffer.data(), chunk_size, name);
        ggml_backend_tensor_set(tensor, reader->buffer.data(), copied, chunk_size);
        copied += chunk_size;
    }

    reader->copied_tensors += 1;
}

std::vector<llama_token> tokenize_text(
    const llama_vocab * vocab,
    const char * text) {
    const int32_t text_len = static_cast<int32_t>(std::strlen(text));
    int32_t token_count = llama_tokenize(vocab, text, text_len, nullptr, 0, true, true);
    if (token_count == INT32_MIN) {
        throw std::runtime_error("eval text tokenization overflowed");
    }
    if (token_count < 0) {
        token_count = -token_count;
    }
    if (token_count <= 0) {
        return {};
    }

    std::vector<llama_token> tokens(static_cast<size_t>(token_count));
    const int32_t actual_tokens = llama_tokenize(
        vocab,
        text,
        text_len,
        tokens.data(),
        token_count,
        true,
        true);
    if (actual_tokens <= 0) {
        return {};
    }

    tokens.resize(static_cast<size_t>(actual_tokens));
    return tokens;
}

struct LlamaPplChunk {
    size_t begin = 0;
    size_t first_scored = 0;
    size_t end = 0;
};

std::vector<LlamaPplChunk> build_llama_ppl_chunks(
    size_t token_count,
    uint32_t context_tokens = LLAMA_PPL_CONTEXT_TOKENS) {
    std::vector<LlamaPplChunk> chunks;
    const size_t context = std::max<size_t>(2, context_tokens);
    if (token_count < 2 * context) {
        return chunks;
    }

    const size_t chunk_count = token_count / context;
    chunks.reserve(chunk_count);
    for (size_t chunk_index = 0; chunk_index < chunk_count; ++chunk_index) {
        const size_t begin = chunk_index * context;
        chunks.push_back(LlamaPplChunk {
            begin,
            begin + context / 2,
            begin + context,
        });
    }

    return chunks;
}

double llama_ppl_uncertainty(double nll_sum, double nll_sq_sum, uint64_t token_count) {
    if (token_count <= 1) {
        return 0.0;
    }

    const double count = static_cast<double>(token_count);
    const double mean_nll = nll_sum / count;
    double variance = (nll_sq_sum / count) - (mean_nll * mean_nll);
    if (variance <= 0.0 || !std::isfinite(variance)) {
        return 0.0;
    }

    variance = std::sqrt(variance / static_cast<double>(token_count - 1));
    return variance * std::exp(std::min(mean_nll, 700.0));
}

double token_nll_from_logits(const float * logits, int32_t n_vocab, llama_token target) {
    if (logits == nullptr) {
        throw std::runtime_error("llama.cpp returned null logits during eval");
    }
    if (target < 0 || target >= n_vocab) {
        throw std::runtime_error("eval target token is outside model vocabulary");
    }

    float max_logit = -std::numeric_limits<float>::infinity();
    for (int32_t i = 0; i < n_vocab; ++i) {
        max_logit = std::max(max_logit, logits[i]);
    }

    double exp_sum = 0.0;
    for (int32_t i = 0; i < n_vocab; ++i) {
        exp_sum += std::exp(static_cast<double>(logits[i] - max_logit));
    }

    const double log_sum_exp = static_cast<double>(max_logit) + std::log(exp_sum);
    return log_sum_exp - static_cast<double>(logits[target]);
}

size_t find_common_token_prefix(const std::vector<std::vector<llama_token>> & sequences) {
    if (sequences.empty()) {
        return 0;
    }

    size_t min_len = sequences.front().size();
    for (const std::vector<llama_token> & sequence : sequences) {
        min_len = std::min(min_len, sequence.size());
    }

    size_t prefix = 0;
    for (; prefix < min_len; ++prefix) {
        const llama_token token = sequences.front()[prefix];
        bool all_same = true;
        for (size_t i = 1; i < sequences.size(); ++i) {
            if (sequences[i][prefix] != token) {
                all_same = false;
                break;
            }
        }
        if (!all_same) {
            break;
        }
    }

    return prefix;
}

double llama_mcq_choice_score(double nll, uint64_t scored_token_count) {
    if (scored_token_count == 0) {
        throw std::runtime_error("standard eval choice produced no scored tokens");
    }
    return -nll / static_cast<double>(scored_token_count);
}

double score_sequence_suffix_nll(
    ModelSession & session,
    const std::vector<llama_token> & sequence,
    size_t first_scored_token,
    uint64_t * out_token_count) {
    const llama_vocab * vocab = llama_model_get_vocab(session.model.get());
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    if (n_vocab <= 0) {
        throw std::runtime_error("model vocabulary is empty");
    }
    if (sequence.size() < 2 || first_scored_token == 0 || first_scored_token >= sequence.size()) {
        throw std::runtime_error("standard eval choice produced no scored tokens");
    }

    reset_session_context(session);
    const int prefix_res = llama_decode(
        session.ctx.get(),
        llama_batch_get_one(
            const_cast<llama_token *>(sequence.data()),
            static_cast<int32_t>(first_scored_token)));
    if (prefix_res != 0) {
        throw std::runtime_error("failed to decode standard eval common prefix");
    }
    llama_synchronize(session.ctx.get());

    double nll = 0.0;
    uint64_t scored_tokens = 0;
    for (size_t target_index = first_scored_token; target_index < sequence.size(); ++target_index) {
        throw_if_recipe_test_cancelled();
        const float * logits = llama_get_logits_ith(session.ctx.get(), -1);
        nll += token_nll_from_logits(logits, n_vocab, sequence[target_index]);
        scored_tokens += 1;

        if (target_index + 1 < sequence.size()) {
            llama_token token = sequence[target_index];
            llama_batch batch = llama_batch_get_one(&token, 1);
            const int decode_result = llama_decode(session.ctx.get(), batch);
            if (decode_result != 0) {
                throw std::runtime_error("failed to decode standard eval choice token");
            }
            llama_synchronize(session.ctx.get());
        }
    }

    if (out_token_count != nullptr) {
        *out_token_count = scored_tokens;
    }
    session.vram.sample();
    return nll;
}

std::vector<StandardEvalSampleScore> score_standard_eval_samples(
    ModelSession & session,
    const ms_standard_eval_sample * samples,
    uint64_t sample_count) {
    std::vector<StandardEvalSampleScore> scores;
    if (samples == nullptr || sample_count == 0) {
        return scores;
    }
    scores.reserve(static_cast<size_t>(sample_count));

    for (uint64_t sample_index = 0; sample_index < sample_count; ++sample_index) {
        throw_if_recipe_test_cancelled();
        const ms_standard_eval_sample & sample = samples[sample_index];
        if (sample.task == nullptr || sample.task[0] == '\0') {
            throw std::runtime_error("standard eval sample task is empty");
        }
        if (sample.prompt == nullptr || sample.prompt[0] == '\0') {
            throw std::runtime_error(std::string("standard eval sample prompt is empty for task: ") + sample.task);
        }
        if (sample.choices == nullptr || sample.choice_count < 2) {
            throw std::runtime_error(std::string("standard eval sample needs at least two choices for task: ") + sample.task);
        }
        if (sample.choice_lengths == nullptr) {
            throw std::runtime_error(std::string("standard eval sample choice lengths are missing for task: ") + sample.task);
        }
        if (sample.gold_index >= sample.choice_count) {
            throw std::runtime_error(std::string("standard eval gold index is outside choices for task: ") + sample.task);
        }

        std::vector<double> choice_scores(static_cast<size_t>(sample.choice_count));
        std::vector<double> choice_nlls(static_cast<size_t>(sample.choice_count));
        std::vector<double> choice_denominators(static_cast<size_t>(sample.choice_count));
        std::vector<std::vector<llama_token>> choice_token_sequences;
        choice_token_sequences.reserve(static_cast<size_t>(sample.choice_count));

        const llama_vocab * vocab = llama_model_get_vocab(session.model.get());
        for (uint64_t choice_index = 0; choice_index < sample.choice_count; ++choice_index) {
            throw_if_recipe_test_cancelled();
            if (sample.choices[choice_index] == nullptr || sample.choices[choice_index][0] == '\0') {
                throw std::runtime_error(std::string("standard eval sample choice is empty for task: ") + sample.task);
            }
            std::string candidate = sample.prompt;
            candidate += sample.choices[choice_index];
            std::vector<llama_token> tokens = tokenize_text(vocab, candidate.c_str());
            if (tokens.size() < 2) {
                throw std::runtime_error(std::string("standard eval sample choice produced too few tokens for task: ") + sample.task);
            }
            choice_token_sequences.push_back(std::move(tokens));
        }

        const size_t common_prefix = find_common_token_prefix(choice_token_sequences);
        if (common_prefix == 0) {
            throw std::runtime_error(std::string("standard eval choices share no prompt prefix for task: ") + sample.task);
        }

        for (uint64_t choice_index = 0; choice_index < sample.choice_count; ++choice_index) {
            throw_if_recipe_test_cancelled();
            uint64_t token_count = 0;
            const double nll = score_sequence_suffix_nll(
                session,
                choice_token_sequences[static_cast<size_t>(choice_index)],
                common_prefix,
                &token_count);
            const double denominator = static_cast<double>(token_count);
            choice_denominators[static_cast<size_t>(choice_index)] = denominator;
            choice_nlls[static_cast<size_t>(choice_index)] = nll;
            choice_scores[static_cast<size_t>(choice_index)] = llama_mcq_choice_score(nll, token_count);
        }

        uint32_t best_index = 0;
        uint32_t second_index = 1;
        if (choice_scores[1] > choice_scores[0]) {
            best_index = 1;
            second_index = 0;
        }
        for (uint64_t choice_index = 2; choice_index < sample.choice_count; ++choice_index) {
            const double score = choice_scores[static_cast<size_t>(choice_index)];
            if (score > choice_scores[best_index]) {
                second_index = best_index;
                best_index = static_cast<uint32_t>(choice_index);
            } else if (score > choice_scores[second_index]) {
                second_index = static_cast<uint32_t>(choice_index);
            }
        }

        StandardEvalSampleScore score = {};
        score.task = sample.task;
        score.sample_index = sample_index;
        score.gold_index = sample.gold_index;
        score.prediction_index = best_index;
        score.correct = best_index == sample.gold_index;
        score.margin = choice_scores[best_index] - choice_scores[second_index];
        score.correct_nll = choice_nlls[static_cast<size_t>(sample.gold_index)];
        score.choice_denominators = std::move(choice_denominators);
        score.choice_nlls = std::move(choice_nlls);
        score.choice_scores = std::move(choice_scores);
        scores.push_back(std::move(score));
    }

    return scores;
}

void copy_task_name(char (& out_task)[64], const std::string & task) {
    std::memset(out_task, 0, 64);
    const size_t copy_len = std::min<size_t>(task.size(), 63);
    std::memcpy(out_task, task.data(), copy_len);
}

bool write_standard_eval_results(
    const std::vector<StandardEvalSampleScore> * baseline_scores,
    const std::vector<StandardEvalSampleScore> & recipe_scores,
    ms_standard_eval_task_result * out_task_results,
    uint64_t task_result_capacity,
    uint64_t * out_task_result_count) {
    if (out_task_result_count == nullptr) {
        fail("standard eval task count output pointer is null");
        return false;
    }

    *out_task_result_count = 0;
    if (recipe_scores.empty()) {
        return true;
    }
    if (out_task_results == nullptr && task_result_capacity > 0) {
        fail("standard eval task result pointer is null");
        return false;
    }
    if (baseline_scores != nullptr && baseline_scores->size() != recipe_scores.size()) {
        fail("baseline and recipe standard eval sample counts do not match");
        return false;
    }

    std::map<std::string, StandardEvalAccumulator> accumulators;
    for (size_t i = 0; i < recipe_scores.size(); ++i) {
        const StandardEvalSampleScore * baseline = baseline_scores == nullptr ? nullptr : &(*baseline_scores)[i];
        const StandardEvalSampleScore & recipe = recipe_scores[i];
        if (baseline != nullptr && baseline->task != recipe.task) {
            fail("baseline and recipe standard eval sample order does not match");
            return false;
        }

        StandardEvalAccumulator & acc = accumulators[recipe.task];
        acc.task = recipe.task;
        acc.sample_count += 1;
        acc.recipe_correct_count += recipe.correct ? 1 : 0;
        acc.recipe_margin_sum += recipe.margin;
        acc.recipe_correct_nll_sum += recipe.correct_nll;

        if (baseline != nullptr) {
            acc.baseline_correct_count += baseline->correct ? 1 : 0;
            acc.baseline_margin_sum += baseline->margin;
            acc.baseline_correct_nll_sum += baseline->correct_nll;
            acc.correct_to_wrong_count += baseline->correct && !recipe.correct ? 1 : 0;
            acc.wrong_to_correct_count += !baseline->correct && recipe.correct ? 1 : 0;
            acc.same_prediction_count += baseline->prediction_index == recipe.prediction_index ? 1 : 0;
        }
    }

    *out_task_result_count = static_cast<uint64_t>(accumulators.size());
    if (task_result_capacity < accumulators.size()) {
        fail(
            "standard eval task result capacity is too small: need "
            + std::to_string(accumulators.size())
            + ", got "
            + std::to_string(task_result_capacity));
        return false;
    }

    size_t out_index = 0;
    for (const auto & entry : accumulators) {
        const StandardEvalAccumulator & acc = entry.second;
        ms_standard_eval_task_result result = {};
        copy_task_name(result.task, acc.task);
        result.sample_count = acc.sample_count;
        result.baseline_correct_count = acc.baseline_correct_count;
        result.recipe_correct_count = acc.recipe_correct_count;
        result.correct_to_wrong_count = acc.correct_to_wrong_count;
        result.wrong_to_correct_count = acc.wrong_to_correct_count;
        result.same_prediction_count = acc.same_prediction_count;
        result.baseline_accuracy = acc.sample_count > 0
            ? static_cast<double>(acc.baseline_correct_count) / static_cast<double>(acc.sample_count)
            : 0.0;
        result.recipe_accuracy = acc.sample_count > 0
            ? static_cast<double>(acc.recipe_correct_count) / static_cast<double>(acc.sample_count)
            : 0.0;
        result.accuracy_delta = result.recipe_accuracy - result.baseline_accuracy;
        result.baseline_avg_margin = acc.sample_count > 0
            ? acc.baseline_margin_sum / static_cast<double>(acc.sample_count)
            : 0.0;
        result.recipe_avg_margin = acc.sample_count > 0
            ? acc.recipe_margin_sum / static_cast<double>(acc.sample_count)
            : 0.0;
        result.margin_delta = result.recipe_avg_margin - result.baseline_avg_margin;
        result.baseline_avg_correct_nll = acc.sample_count > 0
            ? acc.baseline_correct_nll_sum / static_cast<double>(acc.sample_count)
            : 0.0;
        result.recipe_avg_correct_nll = acc.sample_count > 0
            ? acc.recipe_correct_nll_sum / static_cast<double>(acc.sample_count)
            : 0.0;
        out_task_results[out_index++] = result;
    }

    return true;
}

void write_choice_audit_values(
    const std::vector<double> & values,
    double (& out_values)[MS_STANDARD_EVAL_AUDIT_MAX_CHOICES],
    uint32_t choice_count) {
    for (uint32_t i = 0; i < choice_count; ++i) {
        out_values[i] = values[static_cast<size_t>(i)];
    }
}

bool write_standard_eval_sample_audits(
    const std::vector<StandardEvalSampleScore> * baseline_scores,
    const std::vector<StandardEvalSampleScore> & recipe_scores,
    ms_standard_eval_sample_audit * out_sample_audits,
    uint64_t sample_audit_capacity,
    uint64_t * out_sample_audit_count) {
    if (out_sample_audit_count == nullptr) {
        fail("standard eval sample audit count output pointer is null");
        return false;
    }

    *out_sample_audit_count = 0;
    if (recipe_scores.empty() || sample_audit_capacity == 0) {
        return true;
    }
    if (out_sample_audits == nullptr) {
        fail("standard eval sample audit pointer is null");
        return false;
    }
    if (baseline_scores != nullptr && baseline_scores->size() != recipe_scores.size()) {
        fail("baseline and recipe standard eval sample audit counts do not match");
        return false;
    }

    std::vector<size_t> selected;
    selected.reserve(static_cast<size_t>(sample_audit_capacity));
    auto add_index = [&](size_t index) {
        if (selected.size() >= static_cast<size_t>(sample_audit_capacity)) {
            return;
        }
        if (std::find(selected.begin(), selected.end(), index) == selected.end()) {
            selected.push_back(index);
        }
    };

    if (baseline_scores != nullptr) {
        for (size_t i = 0; i < recipe_scores.size(); ++i) {
            const StandardEvalSampleScore & baseline = (*baseline_scores)[i];
            const StandardEvalSampleScore & recipe = recipe_scores[i];
            if (baseline.prediction_index != recipe.prediction_index) {
                add_index(i);
            }
        }
    }
    for (size_t i = 0; i < recipe_scores.size(); ++i) {
        if (!recipe_scores[i].correct) {
            add_index(i);
        }
    }
    for (size_t i = 0; i < recipe_scores.size(); ++i) {
        add_index(i);
    }

    for (size_t out_index = 0; out_index < selected.size(); ++out_index) {
        const size_t score_index = selected[out_index];
        const StandardEvalSampleScore & recipe = recipe_scores[score_index];
        const StandardEvalSampleScore * baseline = baseline_scores == nullptr
            ? nullptr
            : &(*baseline_scores)[score_index];

        if (baseline != nullptr && baseline->task != recipe.task) {
            fail("baseline and recipe standard eval sample audit order does not match");
            return false;
        }
        const size_t source_choice_count = recipe.choice_scores.size();
        const uint32_t choice_count = static_cast<uint32_t>(
            std::min<size_t>(source_choice_count, MS_STANDARD_EVAL_AUDIT_MAX_CHOICES));

        ms_standard_eval_sample_audit audit = {};
        audit.sample_index = recipe.sample_index;
        copy_task_name(audit.task, recipe.task);
        audit.choice_count = choice_count;
        audit.gold_index = recipe.gold_index;
        audit.has_baseline = baseline == nullptr ? 0u : 1u;
        audit.baseline_prediction_index = baseline == nullptr ? 0u : baseline->prediction_index;
        audit.recipe_prediction_index = recipe.prediction_index;
        audit.baseline_correct = baseline != nullptr && baseline->correct ? 1u : 0u;
        audit.recipe_correct = recipe.correct ? 1u : 0u;
        write_choice_audit_values(recipe.choice_denominators, audit.choice_denominators, choice_count);
        write_choice_audit_values(recipe.choice_nlls, audit.recipe_choice_nlls, choice_count);
        write_choice_audit_values(recipe.choice_scores, audit.recipe_choice_scores, choice_count);
        if (baseline != nullptr) {
            write_choice_audit_values(baseline->choice_nlls, audit.baseline_choice_nlls, choice_count);
            write_choice_audit_values(baseline->choice_scores, audit.baseline_choice_scores, choice_count);
        }
        out_sample_audits[out_index] = audit;
    }

    *out_sample_audit_count = static_cast<uint64_t>(selected.size());
    return true;
}

PerplexityScore score_session_perplexity(
    ModelSession & session,
    const char * const * eval_texts,
    uint64_t eval_text_count,
    uint32_t max_eval_tokens) {
    PerplexityScore score = {};
    (void)max_eval_tokens;
    if (eval_texts == nullptr || eval_text_count == 0) {
        throw std::runtime_error("eval text set is empty");
    }

    const llama_vocab * vocab = llama_model_get_vocab(session.model.get());
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    if (n_vocab <= 0) {
        throw std::runtime_error("model vocabulary is empty");
    }
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    const llama_token bos_token = llama_vocab_bos(vocab);

    std::string corpus;
    for (uint64_t text_index = 0; text_index < eval_text_count; ++text_index) {
        const char * text = eval_texts[text_index];
        if (text == nullptr || text[0] == '\0') {
            score.skipped_count += 1;
            continue;
        }

        if (!corpus.empty()) {
            corpus.append("\n\n");
        }
        corpus.append(text);
        score.sample_count += 1;
    }

    if (corpus.empty()) {
        throw std::runtime_error("eval text set is empty");
    }

    std::vector<llama_token> tokens = tokenize_text(vocab, corpus.c_str());
    const std::vector<LlamaPplChunk> chunks = build_llama_ppl_chunks(tokens.size());
    if (chunks.empty()) {
        throw std::runtime_error("eval text set produced too few tokens for llama.cpp-style PPL");
    }

    double nll_squared_sum = 0.0;

    const auto eval_start = std::chrono::steady_clock::now();
    for (const LlamaPplChunk & chunk : chunks) {
        throw_if_recipe_test_cancelled();
        reset_session_context(session);
        for (size_t token_index = chunk.begin; token_index + 1 < chunk.end; ++token_index) {
            throw_if_recipe_test_cancelled();
            llama_token token = tokens[token_index];
            if (add_bos && token_index == chunk.begin) {
                token = bos_token;
            }

            llama_batch batch = llama_batch_get_one(&token, 1);
            const int decode_result = llama_decode(session.ctx.get(), batch);
            if (decode_result != 0) {
                throw std::runtime_error("failed to decode eval token");
            }
            llama_synchronize(session.ctx.get());

            if (token_index >= chunk.first_scored) {
                const size_t target_index = token_index + 1;
                const float * logits = llama_get_logits_ith(session.ctx.get(), -1);
                const double token_nll = token_nll_from_logits(logits, n_vocab, tokens[target_index]);
                score.total_nll += token_nll;
                nll_squared_sum += token_nll * token_nll;
                score.token_count += 1;
            }
        }
        session.vram.sample();
    }
    const auto eval_end = std::chrono::steady_clock::now();

    if (score.token_count == 0) {
        throw std::runtime_error("eval text set produced no scoreable tokens");
    }

    score.eval_ms = elapsed_ms(eval_start, eval_end);
    const double avg_nll = score.total_nll / static_cast<double>(score.token_count);
    score.ppl = std::exp(std::min(avg_nll, 700.0));
    score.ppl_uncertainty = llama_ppl_uncertainty(
        score.total_nll,
        nll_squared_sum,
        score.token_count);
    score.vram_peak_mb = session.vram.peak_mb;
    score.vram_allocated_mb = session.vram.current_mb;
    return score;
}

int32_t decode_prompt_tokens(
    ModelSession & session,
    std::vector<llama_token> & prompt_tokens,
    uint32_t max_generated_tokens,
    const char * error_label) {
    const uint32_t context_tokens = llama_n_ctx(session.ctx.get());
    if (prompt_tokens.size() + static_cast<size_t>(max_generated_tokens) > context_tokens) {
        return fail(
            std::string(error_label)
            + " exceeds context window: prompt tokens="
            + std::to_string(prompt_tokens.size())
            + ", max generated tokens="
            + std::to_string(max_generated_tokens)
            + ", context tokens="
            + std::to_string(context_tokens));
    }

    const uint32_t batch_limit = std::max<uint32_t>(1, llama_n_batch(session.ctx.get()));
    for (size_t offset = 0; offset < prompt_tokens.size(); offset += batch_limit) {
        const size_t remaining = prompt_tokens.size() - offset;
        const size_t chunk_size = std::min<size_t>(remaining, batch_limit);
        llama_batch batch = llama_batch_get_one(
            prompt_tokens.data() + offset,
            static_cast<int32_t>(chunk_size));
        const int decode_result = llama_decode(session.ctx.get(), batch);
        if (decode_result != 0) {
            return fail(std::string("failed to decode ") + error_label);
        }
    }

    return 0;
}

int32_t run_session_benchmark(
    ModelSession & session,
    const char * prompt,
    uint32_t max_tokens,
    ms_baseline_benchmark * out_benchmark) {
    const llama_vocab * vocab = llama_model_get_vocab(session.model.get());
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

    reset_session_context(session);

    const auto prompt_start = std::chrono::steady_clock::now();
    const int prompt_res = decode_prompt_tokens(
        session,
        prompt_tokens,
        max_tokens,
        "generation prompt");
    if (prompt_res != 0) {
        return prompt_res;
    }
    llama_synchronize(session.ctx.get());
    const auto prompt_end = std::chrono::steady_clock::now();
    session.vram.sample();

    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(
        llama_sampler_chain_init(llama_sampler_chain_default_params()),
        llama_sampler_free);
    llama_sampler_chain_add(sampler.get(), llama_sampler_init_greedy());

    uint32_t generated = 0;
    const auto generation_start = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < max_tokens; ++i) {
        throw_if_recipe_test_cancelled();
        llama_token token = llama_sampler_sample(sampler.get(), session.ctx.get(), -1);
        if (llama_vocab_is_eog(vocab, token)) {
            break;
        }

        llama_batch batch = llama_batch_get_one(&token, 1);
        const int gen_res = llama_decode(session.ctx.get(), batch);
        if (gen_res != 0) {
            return fail("failed to decode generated token");
        }
        llama_synchronize(session.ctx.get());
        ++generated;
    }
    const auto generation_end = std::chrono::steady_clock::now();
    session.vram.sample();

    const double prompt_time_ms = elapsed_ms(prompt_start, prompt_end);
    const double generation_time_ms = elapsed_ms(generation_start, generation_end);

    out_benchmark->load_ms = session.load_ms;
    out_benchmark->prompt_eval_ms = prompt_time_ms;
    out_benchmark->generation_ms = generation_time_ms;
    out_benchmark->prompt_eval_tps = prompt_time_ms > 0.0
        ? (static_cast<double>(prompt_tokens.size()) * 1000.0 / prompt_time_ms)
        : 0.0;
    out_benchmark->token_gen_tps = generation_time_ms > 0.0
        ? (static_cast<double>(generated) * 1000.0 / generation_time_ms)
        : 0.0;
    out_benchmark->ttft_ms = prompt_time_ms;
    out_benchmark->vram_peak_mb = session.vram.peak_mb;
    out_benchmark->vram_allocated_mb = session.vram.current_mb;
    out_benchmark->prompt_tokens = static_cast<uint32_t>(prompt_tokens.size());
    out_benchmark->generated_tokens = generated;
    out_benchmark->copied_tensor_count = 0;
    out_benchmark->converted_tensor_count = 0;
    out_benchmark->converted_bytes_before = 0;
    out_benchmark->converted_bytes_after = 0;
    out_benchmark->requested_target_count = 0;
    out_benchmark->verified_target_count = 0;

    return 0;
}

std::string token_to_piece(const llama_vocab * vocab, llama_token token) {
    std::string piece(32, '\0');
    int32_t n_chars = llama_token_to_piece(
        vocab,
        token,
        piece.data(),
        static_cast<int32_t>(piece.size()),
        0,
        false);
    if (n_chars < 0) {
        piece.resize(static_cast<size_t>(-n_chars));
        n_chars = llama_token_to_piece(
            vocab,
            token,
            piece.data(),
            static_cast<int32_t>(piece.size()),
            0,
            false);
    }
    if (n_chars < 0) {
        throw std::runtime_error("failed to decode generated token piece");
    }
    piece.resize(static_cast<size_t>(n_chars));
    return piece;
}

common_chat_params format_chat_prompt_with_template(
    const common_chat_templates * chat_templates,
    const std::vector<std::pair<std::string, std::string>> & messages,
    bool add_generation_prompt,
    const std::map<std::string, std::string> & chat_template_kwargs = {},
    common_reasoning_format reasoning_format = COMMON_REASONING_FORMAT_NONE) {
    if (messages.empty()) {
        throw std::runtime_error("chat completion request must include at least one message");
    }

    if (chat_templates == nullptr) {
        throw std::runtime_error("chat template is not initialized");
    }

    common_chat_templates_inputs inputs;
    inputs.messages.reserve(messages.size());
    inputs.add_generation_prompt = add_generation_prompt;
    inputs.use_jinja = true;
    inputs.reasoning_format = reasoning_format;
    inputs.chat_template_kwargs = chat_template_kwargs;

    const auto enable_thinking = inputs.chat_template_kwargs.find("enable_thinking");
    if (enable_thinking != inputs.chat_template_kwargs.end()) {
        if (enable_thinking->second == "true") {
            inputs.enable_thinking = true;
        } else if (enable_thinking->second == "false") {
            inputs.enable_thinking = false;
        } else if (!enable_thinking->second.empty() && enable_thinking->second[0] == '"') {
            throw std::invalid_argument("invalid type for \"enable_thinking\" (expected boolean, got string)");
        }
    }

    for (const auto & message : messages) {
        if (message.first.empty()) {
            throw std::runtime_error("chat message role must not be empty");
        }
        common_chat_msg chat_message = {};
        chat_message.role = message.first;
        chat_message.content = message.second;
        inputs.messages.push_back(std::move(chat_message));
    }

    return common_chat_templates_apply(chat_templates, inputs);
}

std::map<std::string, std::string> chat_template_kwargs_from_json(const char * json_text) {
    std::map<std::string, std::string> kwargs;
    if (json_text == nullptr || json_text[0] == '\0') {
        return kwargs;
    }

    const nlohmann::ordered_json parsed = nlohmann::ordered_json::parse(json_text);
    if (!parsed.is_object()) {
        throw std::invalid_argument("chat_template_kwargs must be a JSON object");
    }
    for (const auto & item : parsed.items()) {
        kwargs[item.key()] = item.value().dump();
    }
    return kwargs;
}

common_reasoning_format reasoning_format_from_request(const char * value) {
    if (value == nullptr || value[0] == '\0') {
        return COMMON_REASONING_FORMAT_DEEPSEEK;
    }
    return common_reasoning_format_from_name(value);
}

int32_t normalize_context_window_param(int32_t value, uint32_t context_tokens) {
    if (value == -1) {
        return static_cast<int32_t>(std::min<uint32_t>(
            context_tokens,
            static_cast<uint32_t>(std::numeric_limits<int32_t>::max())));
    }
    return value;
}

common_params_sampling common_sampling_from_chat_params(
    const ms_chat_generation_params & params,
    uint32_t context_tokens) {
    common_params_sampling sampling;
    sampling.seed = params.seed;
    sampling.top_k = params.top_k;
    sampling.top_p = static_cast<float>(params.top_p);
    sampling.min_p = static_cast<float>(params.min_p);
    sampling.typ_p = static_cast<float>(params.typical_p);
    sampling.temp = static_cast<float>(params.temperature);
    sampling.penalty_last_n = normalize_context_window_param(params.repeat_last_n, context_tokens);
    sampling.penalty_repeat = static_cast<float>(params.repeat_penalty);
    sampling.penalty_freq = static_cast<float>(params.frequency_penalty);
    sampling.penalty_present = static_cast<float>(params.presence_penalty);
    sampling.dry_multiplier = static_cast<float>(params.dry_multiplier);
    sampling.dry_base = static_cast<float>(params.dry_base);
    sampling.dry_allowed_length = params.dry_allowed_length;
    sampling.dry_penalty_last_n = normalize_context_window_param(params.dry_penalty_last_n, context_tokens);
    return sampling;
}

ms_chat_generation_params default_chat_generation_params(uint32_t max_tokens) {
    ms_chat_generation_params params = {};
    params.max_tokens = max_tokens;
    params.add_generation_prompt = 1;
    params.seed = LLAMA_DEFAULT_SEED;
    params.top_k = 40;
    params.repeat_last_n = 64;
    params.dry_allowed_length = 2;
    params.dry_penalty_last_n = -1;
    params.temperature = 0.8;
    params.top_p = 0.95;
    params.min_p = 0.05;
    params.typical_p = 1.0;
    params.repeat_penalty = 1.0;
    params.frequency_penalty = 0.0;
    params.presence_penalty = 0.0;
    params.dry_multiplier = 0.0;
    params.dry_base = 1.75;
    return params;
}

bool consume_generated_stop_suffix(std::string & generated_text, const std::vector<std::string> & stop_strings) {
    for (const std::string & stop : stop_strings) {
        if (stop.empty() || generated_text.size() < stop.size()) {
            continue;
        }
        const size_t offset = generated_text.size() - stop.size();
        if (generated_text.compare(offset, stop.size(), stop) == 0) {
            generated_text.resize(offset);
            return true;
        }
    }
    return false;
}

std::vector<std::string> collect_stop_strings(const char * const * stop_strings, uint64_t stop_count) {
    std::vector<std::string> stops;
    stops.reserve(static_cast<size_t>(stop_count));
    for (uint64_t i = 0; i < stop_count; ++i) {
        if (stop_strings[i] == nullptr) {
            throw std::runtime_error("chat stop string is null");
        }
        if (stop_strings[i][0] != '\0') {
            stops.emplace_back(stop_strings[i]);
        }
    }
    return stops;
}

std::vector<std::string> merge_stop_strings(
    const std::vector<std::string> & request_stops,
    const std::vector<std::string> & template_stops) {
    std::vector<std::string> stops;
    stops.reserve(request_stops.size() + template_stops.size());
    stops.insert(stops.end(), request_stops.begin(), request_stops.end());
    stops.insert(stops.end(), template_stops.begin(), template_stops.end());
    return stops;
}

ParsedChatOutput parse_generated_chat_output(
    const std::string & generated_text,
    const common_chat_params & chat_params,
    common_reasoning_format reasoning_format,
    bool is_partial) {
    common_chat_parser_params parser_params(chat_params);
    parser_params.reasoning_format = reasoning_format;
    parser_params.reasoning_in_content = false;
    if (!chat_params.parser.empty()) {
        parser_params.parser.load(chat_params.parser);
    }

    try {
        const common_chat_msg parsed = common_chat_parse(generated_text, is_partial, parser_params);
        return ParsedChatOutput{
            parsed.render_content(),
            parsed.reasoning_content,
        };
    } catch (...) {
        return ParsedChatOutput{generated_text, ""};
    }
}

int32_t run_session_generate(
    ModelSession & session,
    const char * prompt,
    const ms_chat_generation_params & params,
    const std::vector<std::string> & stop_strings,
    std::string & generated_text,
    ms_baseline_benchmark * out_benchmark,
    uint32_t & out_finish_reason,
    uint32_t & out_actual_seed) {
    out_finish_reason = MS_CHAT_FINISH_REASON_LENGTH;
    const llama_vocab * vocab = llama_model_get_vocab(session.model.get());
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

    reset_session_context(session);

    const uint32_t context_tokens = llama_n_ctx(session.ctx.get());
    const uint32_t max_tokens = context_generation_room(context_tokens, prompt_tokens.size());
    if (max_tokens == 0) {
        return fail(
            std::string("benchmark prompt exceeds context window: prompt tokens=")
            + std::to_string(prompt_tokens.size())
            + ", context tokens="
            + std::to_string(context_tokens));
    }

    const auto prompt_start = std::chrono::steady_clock::now();
    const int prompt_res = decode_prompt_tokens(
        session,
        prompt_tokens,
        max_tokens,
        "benchmark prompt");
    if (prompt_res != 0) {
        return prompt_res;
    }
    llama_synchronize(session.ctx.get());
    const auto prompt_end = std::chrono::steady_clock::now();
    session.vram.sample();

    common_params_sampling sampling_params = common_sampling_from_chat_params(
        params,
        llama_n_ctx(session.ctx.get()));
    common_sampler_ptr sampler(common_sampler_init(session.model.get(), sampling_params));
    if (!sampler) {
        return fail("failed to initialize llama.cpp common sampler");
    }
    out_actual_seed = common_sampler_get_seed(sampler.get());
    for (const llama_token token : prompt_tokens) {
        common_sampler_accept(sampler.get(), token, false);
    }

    uint32_t generated = 0;
    generated_text.clear();
    const auto generation_start = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < max_tokens; ++i) {
        throw_if_recipe_test_cancelled();
        llama_token token = common_sampler_sample(sampler.get(), session.ctx.get(), -1);
        if (llama_vocab_is_eog(vocab, token)) {
            out_finish_reason = MS_CHAT_FINISH_REASON_EOS;
            break;
        }

        generated_text += token_to_piece(vocab, token);
        common_sampler_accept(sampler.get(), token, true);
        ++generated;
        if (consume_generated_stop_suffix(generated_text, stop_strings)) {
            out_finish_reason = MS_CHAT_FINISH_REASON_STOP;
            break;
        }

        llama_batch batch = llama_batch_get_one(&token, 1);
        const int gen_res = llama_decode(session.ctx.get(), batch);
        if (gen_res != 0) {
            return fail("failed to decode generated token");
        }
        llama_synchronize(session.ctx.get());
    }
    const auto generation_end = std::chrono::steady_clock::now();
    session.vram.sample();

    const double prompt_time_ms = elapsed_ms(prompt_start, prompt_end);
    const double generation_time_ms = elapsed_ms(generation_start, generation_end);

    out_benchmark->load_ms = session.load_ms;
    out_benchmark->prompt_eval_ms = prompt_time_ms;
    out_benchmark->generation_ms = generation_time_ms;
    out_benchmark->prompt_eval_tps = prompt_time_ms > 0.0
        ? (static_cast<double>(prompt_tokens.size()) * 1000.0 / prompt_time_ms)
        : 0.0;
    out_benchmark->token_gen_tps = generation_time_ms > 0.0
        ? (static_cast<double>(generated) * 1000.0 / generation_time_ms)
        : 0.0;
    out_benchmark->ttft_ms = prompt_time_ms;
    out_benchmark->vram_peak_mb = session.vram.peak_mb;
    out_benchmark->vram_allocated_mb = session.vram.current_mb;
    out_benchmark->prompt_tokens = static_cast<uint32_t>(prompt_tokens.size());
    out_benchmark->generated_tokens = generated;
    out_benchmark->copied_tensor_count = 0;
    out_benchmark->converted_tensor_count = 0;
    out_benchmark->converted_bytes_before = 0;
    out_benchmark->converted_bytes_after = 0;
    out_benchmark->requested_target_count = 0;
    out_benchmark->verified_target_count = 0;

    return 0;
}

bool has_prefix(const std::string & value, const std::string & prefix) {
    return value.size() >= prefix.size()
        && value.compare(0, prefix.size(), prefix) == 0;
}

int32_t emit_chat_stream_delta(
    ms_chat_stream_callback callback,
    void * user_data,
    const std::string & visible_delta,
    const std::string & reasoning_delta) {
    if (visible_delta.empty() && reasoning_delta.empty()) {
        return 0;
    }
    const int32_t status = callback(
        visible_delta.c_str(),
        reasoning_delta.c_str(),
        user_data);
    if (status != 0) {
        return fail("chat stream callback aborted");
    }
    return 0;
}

int32_t emit_chat_stream_parse_delta(
    const std::string & generated_text,
    const common_chat_params & chat_params,
    common_reasoning_format reasoning_format,
    bool is_partial,
    std::string & emitted_visible_text,
    std::string & emitted_reasoning_text,
    ms_chat_stream_callback callback,
    void * user_data) {
    const ParsedChatOutput parsed = parse_generated_chat_output(
        generated_text,
        chat_params,
        reasoning_format,
        is_partial);

    std::string visible_delta;
    if (has_prefix(parsed.visible_text, emitted_visible_text)) {
        visible_delta = parsed.visible_text.substr(emitted_visible_text.size());
    }

    std::string reasoning_delta;
    if (has_prefix(parsed.reasoning_text, emitted_reasoning_text)) {
        reasoning_delta = parsed.reasoning_text.substr(emitted_reasoning_text.size());
    }

    const int32_t result = emit_chat_stream_delta(
        callback,
        user_data,
        visible_delta,
        reasoning_delta);
    if (result != 0) {
        return result;
    }

    if (!visible_delta.empty()) {
        emitted_visible_text = parsed.visible_text;
    }
    if (!reasoning_delta.empty()) {
        emitted_reasoning_text = parsed.reasoning_text;
    }
    return 0;
}

constexpr size_t MAX_MULTIMODAL_IMAGE_BYTES = 20 * 1024 * 1024;

struct MtmdBitmapDeleter {
    void operator()(mtmd_bitmap * bitmap) const {
        mtmd_bitmap_free(bitmap);
    }
};

struct MtmdChunksDeleter {
    void operator()(mtmd_input_chunks * chunks) const {
        mtmd_input_chunks_free(chunks);
    }
};

struct MultimodalPromptEvaluation {
    std::vector<llama_token> sampler_tokens;
    uint32_t prompt_tokens = 0;
};

std::vector<unsigned char> decode_image_data_url(const char * image_data_url) {
    if (image_data_url == nullptr) {
        throw std::runtime_error("image data URL is null");
    }

    const std::string value(image_data_url);
    constexpr std::string_view prefix = "data:image/";
    if (value.compare(0, prefix.size(), prefix) != 0) {
        throw std::runtime_error("image_url must be a base64 data:image URL");
    }

    const size_t separator = value.find(";base64,");
    if (separator == std::string::npos || separator <= prefix.size()) {
        throw std::runtime_error("image_url must use base64 encoding");
    }

    const std::string encoded = value.substr(separator + std::strlen(";base64,"));
    if (encoded.empty()) {
        throw std::runtime_error("image_url has no image data");
    }
    if (encoded.size() > MAX_MULTIMODAL_IMAGE_BYTES * 2) {
        throw std::runtime_error("image_url exceeds the 20 MiB limit");
    }

    const std::string decoded = base64::decode(encoded);
    if (decoded.empty()) {
        throw std::runtime_error("image_url decoded to an empty image");
    }
    if (decoded.size() > MAX_MULTIMODAL_IMAGE_BYTES) {
        throw std::runtime_error("image_url exceeds the 20 MiB limit");
    }

    return std::vector<unsigned char>(decoded.begin(), decoded.end());
}

int32_t evaluate_multimodal_prompt(
    ModelSession & session,
    const char * prompt,
    const char * const * image_data_urls,
    uint64_t image_count,
    uint32_t & max_tokens,
    MultimodalPromptEvaluation & evaluation) {
    if (session.mctx == nullptr) {
        return fail("image input requires an MMPROJ GGUF loaded in the MMPROJ section");
    }
    if (image_data_urls == nullptr || image_count == 0) {
        return fail("multimodal generation requires at least one image");
    }

    std::vector<std::unique_ptr<mtmd_bitmap, MtmdBitmapDeleter>> bitmaps;
    bitmaps.reserve(static_cast<size_t>(image_count));
    for (uint64_t i = 0; i < image_count; ++i) {
        const std::vector<unsigned char> image = decode_image_data_url(image_data_urls[i]);
        mtmd_bitmap * bitmap = mtmd_helper_bitmap_init_from_buf(
            session.mctx.get(), image.data(), image.size());
        if (bitmap == nullptr) {
            return fail("failed to decode image input");
        }
        bitmaps.emplace_back(bitmap);
    }

    std::vector<const mtmd_bitmap *> bitmap_ptrs;
    bitmap_ptrs.reserve(bitmaps.size());
    for (const auto & bitmap : bitmaps) {
        bitmap_ptrs.push_back(bitmap.get());
    }

    std::unique_ptr<mtmd_input_chunks, MtmdChunksDeleter> chunks(
        mtmd_input_chunks_init());
    if (chunks == nullptr) {
        return fail("failed to initialize multimodal input chunks");
    }

    const mtmd_input_text input = {prompt, true, true};
    const int32_t tokenize_result = mtmd_tokenize(
        session.mctx.get(),
        chunks.get(),
        &input,
        bitmap_ptrs.data(),
        bitmap_ptrs.size());
    if (tokenize_result != 0) {
        return fail("failed to tokenize multimodal prompt");
    }

    const size_t prompt_tokens = mtmd_helper_get_n_tokens(chunks.get());
    if (prompt_tokens == 0 || prompt_tokens > std::numeric_limits<uint32_t>::max()) {
        return fail("multimodal prompt produced no tokens");
    }

    const uint32_t context_tokens = llama_n_ctx(session.ctx.get());
    max_tokens = context_generation_room(context_tokens, prompt_tokens);
    if (max_tokens == 0) {
        return fail(
            std::string("multimodal prompt exceeds context window: prompt tokens=")
            + std::to_string(prompt_tokens)
            + ", context tokens="
            + std::to_string(context_tokens));
    }

    reset_session_context(session);
    llama_pos new_n_past = 0;
    const int32_t eval_result = mtmd_helper_eval_chunks(
        session.mctx.get(),
        session.ctx.get(),
        chunks.get(),
        0,
        0,
        static_cast<int32_t>(std::min<uint32_t>(512, context_tokens)),
        true,
        &new_n_past);
    if (eval_result != 0) {
        return fail("failed to evaluate multimodal prompt");
    }
    llama_synchronize(session.ctx.get());

    evaluation.sampler_tokens.clear();
    const size_t chunk_count = mtmd_input_chunks_size(chunks.get());
    for (size_t index = 0; index < chunk_count; ++index) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks.get(), index);
        if (mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_TEXT) {
            continue;
        }
        size_t token_count = 0;
        const llama_token * tokens = mtmd_input_chunk_get_tokens_text(chunk, &token_count);
        if (tokens != nullptr && token_count > 0) {
            evaluation.sampler_tokens.insert(
                evaluation.sampler_tokens.end(), tokens, tokens + token_count);
        }
    }
    evaluation.prompt_tokens = static_cast<uint32_t>(prompt_tokens);
    return 0;
}

int32_t run_session_generate_multimodal_stream(
    ModelSession & session,
    const char * prompt,
    const char * const * image_data_urls,
    uint64_t image_count,
    const ms_chat_generation_params & params,
    const std::vector<std::string> & stop_strings,
    const common_chat_params & chat_params,
    common_reasoning_format reasoning_format,
    ms_chat_stream_callback callback,
    void * user_data,
    ms_baseline_benchmark * out_benchmark,
    uint32_t & out_finish_reason,
    uint32_t & out_actual_seed) {
    out_finish_reason = MS_CHAT_FINISH_REASON_LENGTH;
    const llama_vocab * vocab = llama_model_get_vocab(session.model.get());
    uint32_t max_tokens = 0;
    MultimodalPromptEvaluation evaluation = {};

    const auto prompt_start = std::chrono::steady_clock::now();
    const int prompt_res = evaluate_multimodal_prompt(
        session,
        prompt,
        image_data_urls,
        image_count,
        max_tokens,
        evaluation);
    if (prompt_res != 0) {
        return prompt_res;
    }
    const auto prompt_end = std::chrono::steady_clock::now();
    session.vram.sample();

    common_params_sampling sampling_params = common_sampling_from_chat_params(
        params,
        llama_n_ctx(session.ctx.get()));
    common_sampler_ptr sampler(common_sampler_init(session.model.get(), sampling_params));
    if (!sampler) {
        return fail("failed to initialize llama.cpp common sampler");
    }
    out_actual_seed = common_sampler_get_seed(sampler.get());
    for (const llama_token token : evaluation.sampler_tokens) {
        common_sampler_accept(sampler.get(), token, false);
    }

    uint32_t generated = 0;
    std::string generated_text;
    std::string emitted_visible_text;
    std::string emitted_reasoning_text;
    const auto generation_start = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < max_tokens; ++i) {
        throw_if_recipe_test_cancelled();
        llama_token token = common_sampler_sample(sampler.get(), session.ctx.get(), -1);
        if (llama_vocab_is_eog(vocab, token)) {
            out_finish_reason = MS_CHAT_FINISH_REASON_EOS;
            break;
        }

        generated_text += token_to_piece(vocab, token);
        common_sampler_accept(sampler.get(), token, true);
        ++generated;
        if (consume_generated_stop_suffix(generated_text, stop_strings)) {
            out_finish_reason = MS_CHAT_FINISH_REASON_STOP;
        }

        const int stream_res = emit_chat_stream_parse_delta(
            generated_text,
            chat_params,
            reasoning_format,
            true,
            emitted_visible_text,
            emitted_reasoning_text,
            callback,
            user_data);
        if (stream_res != 0) {
            return stream_res;
        }

        if (out_finish_reason == MS_CHAT_FINISH_REASON_STOP) {
            break;
        }

        llama_batch batch = llama_batch_get_one(&token, 1);
        if (llama_decode(session.ctx.get(), batch) != 0) {
            return fail("failed to decode generated token");
        }
        llama_synchronize(session.ctx.get());
    }

    const int final_stream_res = emit_chat_stream_parse_delta(
        generated_text,
        chat_params,
        reasoning_format,
        out_finish_reason == MS_CHAT_FINISH_REASON_LENGTH,
        emitted_visible_text,
        emitted_reasoning_text,
        callback,
        user_data);
    if (final_stream_res != 0) {
        return final_stream_res;
    }

    const auto generation_end = std::chrono::steady_clock::now();
    session.vram.sample();
    const double prompt_time_ms = elapsed_ms(prompt_start, prompt_end);
    const double generation_time_ms = elapsed_ms(generation_start, generation_end);

    out_benchmark->load_ms = session.load_ms;
    out_benchmark->prompt_eval_ms = prompt_time_ms;
    out_benchmark->generation_ms = generation_time_ms;
    out_benchmark->prompt_eval_tps = prompt_time_ms > 0.0
        ? (static_cast<double>(evaluation.prompt_tokens) * 1000.0 / prompt_time_ms)
        : 0.0;
    out_benchmark->token_gen_tps = generation_time_ms > 0.0
        ? (static_cast<double>(generated) * 1000.0 / generation_time_ms)
        : 0.0;
    out_benchmark->ttft_ms = prompt_time_ms;
    out_benchmark->vram_peak_mb = session.vram.peak_mb;
    out_benchmark->vram_allocated_mb = session.vram.current_mb;
    out_benchmark->prompt_tokens = evaluation.prompt_tokens;
    out_benchmark->generated_tokens = generated;
    out_benchmark->copied_tensor_count = 0;
    out_benchmark->converted_tensor_count = 0;
    out_benchmark->converted_bytes_before = 0;
    out_benchmark->converted_bytes_after = 0;
    out_benchmark->requested_target_count = 0;
    out_benchmark->verified_target_count = 0;
    return 0;
}

int32_t run_session_generate_stream(
    ModelSession & session,
    const char * prompt,
    const ms_chat_generation_params & params,
    const std::vector<std::string> & stop_strings,
    const common_chat_params & chat_params,
    common_reasoning_format reasoning_format,
    ms_chat_stream_callback callback,
    void * user_data,
    ms_baseline_benchmark * out_benchmark,
    uint32_t & out_finish_reason,
    uint32_t & out_actual_seed) {
    out_finish_reason = MS_CHAT_FINISH_REASON_LENGTH;
    const llama_vocab * vocab = llama_model_get_vocab(session.model.get());
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

    reset_session_context(session);

    const uint32_t context_tokens = llama_n_ctx(session.ctx.get());
    const uint32_t max_tokens = context_generation_room(context_tokens, prompt_tokens.size());
    if (max_tokens == 0) {
        return fail(
            std::string("benchmark prompt exceeds context window: prompt tokens=")
            + std::to_string(prompt_tokens.size())
            + ", context tokens="
            + std::to_string(context_tokens));
    }

    const auto prompt_start = std::chrono::steady_clock::now();
    const int prompt_res = decode_prompt_tokens(
        session,
        prompt_tokens,
        max_tokens,
        "benchmark prompt");
    if (prompt_res != 0) {
        return prompt_res;
    }
    llama_synchronize(session.ctx.get());
    const auto prompt_end = std::chrono::steady_clock::now();
    session.vram.sample();

    common_params_sampling sampling_params = common_sampling_from_chat_params(
        params,
        llama_n_ctx(session.ctx.get()));
    common_sampler_ptr sampler(common_sampler_init(session.model.get(), sampling_params));
    if (!sampler) {
        return fail("failed to initialize llama.cpp common sampler");
    }
    out_actual_seed = common_sampler_get_seed(sampler.get());
    for (const llama_token token : prompt_tokens) {
        common_sampler_accept(sampler.get(), token, false);
    }

    uint32_t generated = 0;
    std::string generated_text;
    std::string emitted_visible_text;
    std::string emitted_reasoning_text;
    const auto generation_start = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < max_tokens; ++i) {
        throw_if_recipe_test_cancelled();
        llama_token token = common_sampler_sample(sampler.get(), session.ctx.get(), -1);
        if (llama_vocab_is_eog(vocab, token)) {
            out_finish_reason = MS_CHAT_FINISH_REASON_EOS;
            break;
        }

        generated_text += token_to_piece(vocab, token);
        common_sampler_accept(sampler.get(), token, true);
        ++generated;
        if (consume_generated_stop_suffix(generated_text, stop_strings)) {
            out_finish_reason = MS_CHAT_FINISH_REASON_STOP;
        }

        const int stream_res = emit_chat_stream_parse_delta(
            generated_text,
            chat_params,
            reasoning_format,
            true,
            emitted_visible_text,
            emitted_reasoning_text,
            callback,
            user_data);
        if (stream_res != 0) {
            return stream_res;
        }

        if (out_finish_reason == MS_CHAT_FINISH_REASON_STOP) {
            break;
        }

        llama_batch batch = llama_batch_get_one(&token, 1);
        const int gen_res = llama_decode(session.ctx.get(), batch);
        if (gen_res != 0) {
            return fail("failed to decode generated token");
        }
        llama_synchronize(session.ctx.get());
    }

    const int final_stream_res = emit_chat_stream_parse_delta(
        generated_text,
        chat_params,
        reasoning_format,
        out_finish_reason == MS_CHAT_FINISH_REASON_LENGTH,
        emitted_visible_text,
        emitted_reasoning_text,
        callback,
        user_data);
    if (final_stream_res != 0) {
        return final_stream_res;
    }

    const auto generation_end = std::chrono::steady_clock::now();
    session.vram.sample();

    const double prompt_time_ms = elapsed_ms(prompt_start, prompt_end);
    const double generation_time_ms = elapsed_ms(generation_start, generation_end);

    out_benchmark->load_ms = session.load_ms;
    out_benchmark->prompt_eval_ms = prompt_time_ms;
    out_benchmark->generation_ms = generation_time_ms;
    out_benchmark->prompt_eval_tps = prompt_time_ms > 0.0
        ? (static_cast<double>(prompt_tokens.size()) * 1000.0 / prompt_time_ms)
        : 0.0;
    out_benchmark->token_gen_tps = generation_time_ms > 0.0
        ? (static_cast<double>(generated) * 1000.0 / generation_time_ms)
        : 0.0;
    out_benchmark->ttft_ms = prompt_time_ms;
    out_benchmark->vram_peak_mb = session.vram.peak_mb;
    out_benchmark->vram_allocated_mb = session.vram.current_mb;
    out_benchmark->prompt_tokens = static_cast<uint32_t>(prompt_tokens.size());
    out_benchmark->generated_tokens = generated;
    out_benchmark->copied_tensor_count = 0;
    out_benchmark->converted_tensor_count = 0;
    out_benchmark->converted_bytes_before = 0;
    out_benchmark->converted_bytes_after = 0;
    out_benchmark->requested_target_count = 0;
    out_benchmark->verified_target_count = 0;

    return 0;
}

} // namespace

struct ms_runtime_chat_session {
    std::unique_ptr<ModelSession> session;
    uint64_t model_load_count = 0;
    uint64_t completion_count = 0;
};

extern "C" {

const char * ms_runtime_version(void) {
    return "model-surgery-runtime/0.2 chat-abi=2";
}

const char * ms_runtime_llama_system_info(void) {
    return llama_print_system_info();
}

const char * ms_runtime_last_error(void) {
    return last_error.c_str();
}

void ms_runtime_reset_recipe_test_cancel(void) {
    recipe_test_cancel_flag.store(false, std::memory_order_relaxed);
}

void ms_runtime_cancel_recipe_test(void) {
    recipe_test_cancel_flag.store(true, std::memory_order_relaxed);
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

int32_t ms_runtime_preview_tensor_values(
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
    uint64_t * out_total_cols) {
    clear_error();

    if (path == nullptr || path[0] == '\0') {
        return fail("GGUF path is empty");
    }
    if (tensor_name == nullptr || tensor_name[0] == '\0') {
        return fail("tensor name is empty");
    }
    if (out_values == nullptr && value_capacity > 0) {
        return fail("tensor preview output pointer is null");
    }
    if (out_rows == nullptr || out_cols == nullptr || out_total_rows == nullptr || out_total_cols == nullptr) {
        return fail("tensor preview metadata output pointer is null");
    }

    try {
        gguf_init_params params = {};
        params.no_alloc = true;
        params.ctx = nullptr;

        std::unique_ptr<gguf_context, decltype(&gguf_free)> ctx(
            gguf_init_from_file(path, params),
            gguf_free);
        if (ctx == nullptr) {
            return fail(std::string("failed to open GGUF: ") + path);
        }

        const int64_t tensor_id = gguf_find_tensor(ctx.get(), tensor_name);
        if (tensor_id < 0) {
            return fail(std::string("GGUF is missing tensor: ") + tensor_name);
        }

        const int64_t cols = gguf_get_tensor_ne(ctx.get(), tensor_id, 0);
        const int64_t row_planes =
            gguf_get_tensor_ne(ctx.get(), tensor_id, 1)
            * gguf_get_tensor_ne(ctx.get(), tensor_id, 2)
            * gguf_get_tensor_ne(ctx.get(), tensor_id, 3);
        if (cols <= 0 || row_planes <= 0) {
            return fail(std::string("tensor has invalid shape: ") + tensor_name);
        }

        *out_total_cols = static_cast<uint64_t>(cols);
        *out_total_rows = static_cast<uint64_t>(row_planes);
        *out_rows = 0;
        *out_cols = 0;

        if (row_offset >= static_cast<uint64_t>(row_planes) || col_offset >= static_cast<uint64_t>(cols)) {
            return 0;
        }

        const uint64_t available_rows = static_cast<uint64_t>(row_planes) - row_offset;
        const uint64_t available_cols = static_cast<uint64_t>(cols) - col_offset;
        const uint64_t rows_to_read = std::min(row_count, available_rows);
        const uint64_t cols_to_read = std::min(col_count, available_cols);
        if (rows_to_read == 0 || cols_to_read == 0) {
            return 0;
        }
        if (rows_to_read > std::numeric_limits<uint64_t>::max() / cols_to_read) {
            return fail("tensor preview window is too large");
        }
        if (rows_to_read * cols_to_read > value_capacity) {
            return fail("tensor preview output buffer is too small");
        }

        const ggml_type current_type = gguf_get_tensor_type(ctx.get(), tensor_id);
        if (current_type != GGML_TYPE_F32) {
            const struct ggml_type_traits * traits = ggml_get_type_traits(current_type);
            if (traits == nullptr || traits->to_float == nullptr) {
                return fail(
                    std::string("unsupported tensor quant for preview: ")
                    + tensor_name + " " + display_quant_type(current_type));
            }
        }

        const size_t row_size = ggml_row_size(current_type, cols);
        const size_t tensor_size = gguf_get_tensor_size(ctx.get(), tensor_id);
        if (tensor_size != row_size * static_cast<size_t>(row_planes)) {
            return fail(std::string("tensor size does not match row layout for: ") + tensor_name);
        }

        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            return fail(std::string("failed to open GGUF data: ") + path);
        }

        UserCopyTensorReader reader;
        reader.file = std::move(file);
        reader.source_metadata = ctx.get();
        reader.data_offset = gguf_get_data_offset(ctx.get());
        reader.source_row_buffer.resize(row_size);
        reader.f32_row_buffer.resize(static_cast<size_t>(cols));

        const size_t tensor_offset = reader.data_offset + gguf_get_tensor_offset(ctx.get(), tensor_id);
        for (uint64_t row = 0; row < rows_to_read; ++row) {
            const uint64_t source_row = row_offset + row;
            read_exact_at(
                &reader,
                tensor_offset + static_cast<size_t>(source_row) * row_size,
                reader.source_row_buffer.data(),
                row_size,
                tensor_name);
            decode_source_row_to_f32(
                reader.source_row_buffer.data(),
                current_type,
                cols,
                reader.f32_row_buffer.data(),
                tensor_name);
            const size_t target_offset = static_cast<size_t>(row * cols_to_read);
            const size_t source_col_offset = static_cast<size_t>(col_offset);
            std::memcpy(
                out_values + target_offset,
                reader.f32_row_buffer.data() + source_col_offset,
                static_cast<size_t>(cols_to_read) * sizeof(float));
        }

        *out_rows = rows_to_read;
        *out_cols = cols_to_read;
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
            throw_if_recipe_test_cancelled();
            analysis.current_size_bytes += static_cast<uint64_t>(gguf_get_tensor_size(ctx, i));
        }

        for (uint64_t i = 0; i < target_count; ++i) {
            throw_if_recipe_test_cancelled();
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
                if (!supports_recipe_conversion(target.name, current_type, target_type)) {
                    analysis.unsupported_count += 1;
                }
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
        std::unique_ptr<ModelSession> session = open_baseline_session(path, session_context_tokens(0));
        return run_session_benchmark(*session, prompt, max_tokens, out_benchmark);
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
        std::unique_ptr<ModelSession> session = open_user_copy_session(path, session_context_tokens(0));
        const int32_t result = run_session_benchmark(*session, prompt, max_tokens, out_benchmark);
        if (result == 0) {
            out_benchmark->copied_tensor_count = session->copied_tensors;
            out_benchmark->converted_tensor_count = session->converted_tensors;
            out_benchmark->converted_bytes_before = session->converted_bytes_before;
            out_benchmark->converted_bytes_after = session->converted_bytes_after;
            out_benchmark->requested_target_count = session->requested_target_count;
            out_benchmark->verified_target_count = session->verified_target_count;
        }
        return result;
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

        std::unique_ptr<ModelSession> session = open_recipe_session(
            path,
            std::move(target_types),
            session_context_tokens(0));
        if (session == nullptr) {
            return -1;
        }
        const int32_t result = run_session_benchmark(*session, prompt, max_tokens, out_benchmark);
        if (result == 0) {
            out_benchmark->copied_tensor_count = session->copied_tensors;
            out_benchmark->converted_tensor_count = session->converted_tensors;
            out_benchmark->converted_bytes_before = session->converted_bytes_before;
            out_benchmark->converted_bytes_after = session->converted_bytes_after;
            out_benchmark->requested_target_count = session->requested_target_count;
            out_benchmark->verified_target_count = session->verified_target_count;
        }
        return result;
    } catch (const std::exception & err) {
        return fail(err.what());
    } catch (...) {
        return fail("unknown native recipe benchmark error");
    }
}

int32_t ms_runtime_generate_recipe(
    const char * path,
    const ms_recipe_tensor_target * targets,
    uint64_t target_count,
    const char * prompt,
    uint32_t max_tokens,
    char * out_text,
    uint64_t out_text_capacity,
    ms_baseline_benchmark * out_benchmark) {
    clear_error();

    if (path == nullptr || path[0] == '\0') {
        return fail("GGUF path is empty");
    }

    if (targets == nullptr && target_count > 0) {
        return fail("recipe target pointer is null");
    }

    if (prompt == nullptr || prompt[0] == '\0') {
        return fail("generation prompt is empty");
    }

    if (out_text == nullptr || out_text_capacity == 0) {
        return fail("generation text output buffer is null or empty");
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

        std::unique_ptr<ModelSession> session = open_recipe_session(
            path,
            std::move(target_types),
            session_context_tokens_for_generation(max_tokens));
        if (session == nullptr) {
            return -1;
        }

        const ms_chat_generation_params params = default_chat_generation_params(max_tokens);
        std::string generated_text;
        uint32_t finish_reason = MS_CHAT_FINISH_REASON_LENGTH;
        uint32_t actual_seed = LLAMA_DEFAULT_SEED;
        const int32_t result = run_session_generate(
            *session,
            prompt,
            params,
            std::vector<std::string>(),
            generated_text,
            out_benchmark,
            finish_reason,
            actual_seed);
        if (result != 0) {
            return result;
        }

        if (generated_text.size() + 1 > out_text_capacity) {
            return fail("generated text output buffer is too small");
        }

        std::memcpy(out_text, generated_text.c_str(), generated_text.size() + 1);
        out_benchmark->copied_tensor_count = session->copied_tensors;
        out_benchmark->converted_tensor_count = session->converted_tensors;
        out_benchmark->converted_bytes_before = session->converted_bytes_before;
        out_benchmark->converted_bytes_after = session->converted_bytes_after;
        out_benchmark->requested_target_count = session->requested_target_count;
        out_benchmark->verified_target_count = session->verified_target_count;
        return 0;
    } catch (const std::exception & err) {
        return fail(err.what());
    } catch (...) {
        return fail("unknown native recipe generation error");
    }
}

int32_t ms_runtime_generate_recipe_chat(
    const char * path,
    const ms_recipe_tensor_target * targets,
    uint64_t target_count,
    const ms_chat_message * messages,
    uint64_t message_count,
    uint32_t max_tokens,
    char * out_text,
    uint64_t out_text_capacity,
    ms_baseline_benchmark * out_benchmark) {
    clear_error();

    if (path == nullptr || path[0] == '\0') {
        return fail("GGUF path is empty");
    }

    if (targets == nullptr && target_count > 0) {
        return fail("recipe target pointer is null");
    }

    if (messages == nullptr || message_count == 0) {
        return fail("chat message set is empty");
    }

    if (out_text == nullptr || out_text_capacity == 0) {
        return fail("generation text output buffer is null or empty");
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

        std::unique_ptr<ModelSession> session = open_recipe_session(
            path,
            std::move(target_types),
            session_context_tokens_for_generation(max_tokens));
        if (session == nullptr) {
            return -1;
        }

        std::vector<std::pair<std::string, std::string>> chat_messages;
        chat_messages.reserve(static_cast<size_t>(message_count));
        for (uint64_t i = 0; i < message_count; ++i) {
            if (messages[i].role == nullptr || messages[i].role[0] == '\0') {
                return fail("chat message role is empty");
            }
            if (messages[i].content == nullptr) {
                return fail("chat message content is null");
            }
            chat_messages.push_back({messages[i].role, messages[i].content});
        }

        const common_chat_params chat_params = format_chat_prompt_with_template(
            session->chat_templates.get(),
            chat_messages,
            true);

        const ms_chat_generation_params params = default_chat_generation_params(max_tokens);
        std::string generated_text;
        uint32_t finish_reason = MS_CHAT_FINISH_REASON_LENGTH;
        uint32_t actual_seed = LLAMA_DEFAULT_SEED;
        const int32_t result = run_session_generate(
            *session,
            chat_params.prompt.c_str(),
            params,
            chat_params.additional_stops,
            generated_text,
            out_benchmark,
            finish_reason,
            actual_seed);
        if (result != 0) {
            return result;
        }

        if (generated_text.size() + 1 > out_text_capacity) {
            return fail("generated text output buffer is too small");
        }

        std::memcpy(out_text, generated_text.c_str(), generated_text.size() + 1);
        out_benchmark->copied_tensor_count = session->copied_tensors;
        out_benchmark->converted_tensor_count = session->converted_tensors;
        out_benchmark->converted_bytes_before = session->converted_bytes_before;
        out_benchmark->converted_bytes_after = session->converted_bytes_after;
        out_benchmark->requested_target_count = session->requested_target_count;
        out_benchmark->verified_target_count = session->verified_target_count;
        return 0;
    } catch (const std::exception & err) {
        return fail(err.what());
    } catch (...) {
        return fail("unknown native recipe chat generation error");
    }
}

int32_t open_recipe_chat_session_impl(
    const char * path,
    const char * projector_path,
    const ms_recipe_tensor_target * targets,
    uint64_t target_count,
    uint32_t context_tokens,
    const RuntimeLogSink * log,
    ms_runtime_chat_session ** out_session) {
    clear_error();

    if (path == nullptr || path[0] == '\0') {
        return fail("GGUF path is empty");
    }

    if (targets == nullptr && target_count > 0) {
        return fail("recipe target pointer is null");
    }

    if (out_session == nullptr) {
        return fail("chat session output pointer is null");
    }

    *out_session = nullptr;

    try {
        ensure_backend_initialized();
        std::unordered_map<std::string, ggml_type> target_types;
        if (!build_recipe_targets(targets, target_count, target_types)) {
            return -1;
        }

        std::unique_ptr<ModelSession> model_session = open_recipe_session(
            path,
            std::move(target_types),
            context_tokens,
            log);
        if (model_session == nullptr) {
            return -1;
        }
        if (projector_path != nullptr && projector_path[0] != '\0') {
            if (log != nullptr) {
                log->emit("Native runtime: loading MMPROJ vision projector");
            }
            const mtmd_context_params projector_params = mtmd_context_params_default();
            model_session->mctx.reset(mtmd_init_from_file(
                projector_path,
                model_session->model.get(),
                projector_params));
            if (model_session->mctx == nullptr) {
                throw std::runtime_error(
                    std::string("failed to load MMPROJ vision projector: ") + projector_path);
            }
            if (!mtmd_support_vision(model_session->mctx.get())) {
                throw std::runtime_error("loaded MMPROJ does not support image input");
            }
            if (log != nullptr) {
                log->emit("Native runtime: MMPROJ vision projector ready");
            }
        }

        auto chat_session = std::make_unique<ms_runtime_chat_session>();
        chat_session->model_load_count = 1;
        chat_session->session = std::move(model_session);
        *out_session = chat_session.release();
        return 0;
    } catch (const std::exception & err) {
        return fail(err.what());
    } catch (...) {
        return fail("unknown native recipe chat session open error");
    }
}

int32_t ms_runtime_open_recipe_chat_session(
    const char * path,
    const ms_recipe_tensor_target * targets,
    uint64_t target_count,
    uint32_t context_tokens,
    ms_runtime_chat_session ** out_session) {
    return open_recipe_chat_session_impl(
        path,
        nullptr,
        targets,
        target_count,
        context_tokens,
        nullptr,
        out_session);
}

int32_t ms_runtime_open_recipe_chat_session_with_progress(
    const char * path,
    const ms_recipe_tensor_target * targets,
    uint64_t target_count,
    uint32_t context_tokens,
    ms_runtime_log_callback log_callback,
    void * log_user_data,
    ms_runtime_chat_session ** out_session) {
    const RuntimeLogSink log{log_callback, log_user_data};
    return open_recipe_chat_session_impl(
        path,
        nullptr,
        targets,
        target_count,
        context_tokens,
        log_callback == nullptr ? nullptr : &log,
        out_session);
}

int32_t ms_runtime_open_recipe_chat_session_with_projector_and_progress(
    const char * path,
    const char * projector_path,
    const ms_recipe_tensor_target * targets,
    uint64_t target_count,
    uint32_t context_tokens,
    ms_runtime_log_callback log_callback,
    void * log_user_data,
    ms_runtime_chat_session ** out_session) {
    const RuntimeLogSink log{log_callback, log_user_data};
    return open_recipe_chat_session_impl(
        path,
        projector_path,
        targets,
        target_count,
        context_tokens,
        log_callback == nullptr ? nullptr : &log,
        out_session);
}

void ms_runtime_close_recipe_chat_session(ms_runtime_chat_session * session) {
    delete session;
}

int32_t ms_runtime_generate_recipe_chat_session(
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
    ms_chat_generation_result * out_result) {
    clear_error();

    if (session == nullptr || session->session == nullptr) {
        return fail("recipe chat session is null");
    }

    if (messages == nullptr || message_count == 0) {
        return fail("chat message set is empty");
    }

    if (params == nullptr) {
        return fail("chat generation params are null");
    }

    if (stop_strings == nullptr && stop_count > 0) {
        return fail("chat stop string pointer is null");
    }

    if (out_text == nullptr || out_text_capacity == 0) {
        return fail("generation text output buffer is null or empty");
    }

    if (out_reasoning_text == nullptr || out_reasoning_text_capacity == 0) {
        return fail("generation reasoning output buffer is null or empty");
    }

    if (out_result == nullptr) {
        return fail("chat generation result output pointer is null");
    }

    try {
        std::vector<std::string> request_stops = collect_stop_strings(stop_strings, stop_count);
        const std::map<std::string, std::string> chat_template_kwargs =
            chat_template_kwargs_from_json(chat_template_kwargs_json);
        const common_reasoning_format native_reasoning_format =
            reasoning_format_from_request(reasoning_format);
        std::vector<std::pair<std::string, std::string>> chat_messages;
        chat_messages.reserve(static_cast<size_t>(message_count));
        for (uint64_t i = 0; i < message_count; ++i) {
            if (messages[i].role == nullptr || messages[i].role[0] == '\0') {
                return fail("chat message role is empty");
            }
            if (messages[i].content == nullptr) {
                return fail("chat message content is null");
            }
            chat_messages.push_back({messages[i].role, messages[i].content});
        }

        const common_chat_params chat_params = format_chat_prompt_with_template(
            session->session->chat_templates.get(),
            chat_messages,
            params->add_generation_prompt != 0,
            chat_template_kwargs,
            native_reasoning_format);

        const std::vector<std::string> all_stops = merge_stop_strings(
            request_stops,
            chat_params.additional_stops);
        std::string generated_text;
        uint32_t finish_reason = MS_CHAT_FINISH_REASON_LENGTH;
        uint32_t actual_seed = LLAMA_DEFAULT_SEED;
        ms_baseline_benchmark benchmark = {};
        const int32_t result = run_session_generate(
            *session->session,
            chat_params.prompt.c_str(),
            *params,
            all_stops,
            generated_text,
            &benchmark,
            finish_reason,
            actual_seed);
        if (result != 0) {
            return result;
        }

        const ParsedChatOutput parsed = parse_generated_chat_output(
            generated_text,
            chat_params,
            native_reasoning_format,
            finish_reason == MS_CHAT_FINISH_REASON_LENGTH);

        if (parsed.visible_text.size() + 1 > out_text_capacity) {
            return fail("generated text output buffer is too small");
        }

        if (parsed.reasoning_text.size() + 1 > out_reasoning_text_capacity) {
            return fail("generated reasoning output buffer is too small");
        }

        std::memcpy(out_text, parsed.visible_text.c_str(), parsed.visible_text.size() + 1);
        std::memcpy(
            out_reasoning_text,
            parsed.reasoning_text.c_str(),
            parsed.reasoning_text.size() + 1);
        benchmark.copied_tensor_count = session->session->copied_tensors;
        benchmark.converted_tensor_count = session->session->converted_tensors;
        benchmark.converted_bytes_before = session->session->converted_bytes_before;
        benchmark.converted_bytes_after = session->session->converted_bytes_after;
        benchmark.requested_target_count = session->session->requested_target_count;
        benchmark.verified_target_count = session->session->verified_target_count;
        out_result->benchmark = benchmark;
        out_result->prompt_tokens = benchmark.prompt_tokens;
        out_result->completion_tokens = benchmark.generated_tokens;
        out_result->finish_reason = finish_reason;
        out_result->actual_seed = actual_seed;
        session->completion_count += 1;
        return 0;
    } catch (const std::exception & err) {
        return fail(err.what());
    } catch (...) {
        return fail("unknown native recipe chat session generation error");
    }
}

int32_t ms_runtime_generate_recipe_chat_session_stream(
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
    ms_chat_generation_result * out_result) {
    clear_error();

    if (session == nullptr || session->session == nullptr) {
        return fail("recipe chat session is null");
    }

    if (messages == nullptr || message_count == 0) {
        return fail("chat message set is empty");
    }

    if (params == nullptr) {
        return fail("chat generation params are null");
    }

    if (stop_strings == nullptr && stop_count > 0) {
        return fail("chat stop string pointer is null");
    }

    if (stream_callback == nullptr) {
        return fail("chat stream callback is null");
    }

    if (out_result == nullptr) {
        return fail("chat generation result output pointer is null");
    }

    try {
        std::vector<std::string> request_stops = collect_stop_strings(stop_strings, stop_count);
        const std::map<std::string, std::string> chat_template_kwargs =
            chat_template_kwargs_from_json(chat_template_kwargs_json);
        const common_reasoning_format native_reasoning_format =
            reasoning_format_from_request(reasoning_format);
        std::vector<std::pair<std::string, std::string>> chat_messages;
        chat_messages.reserve(static_cast<size_t>(message_count));
        for (uint64_t i = 0; i < message_count; ++i) {
            if (messages[i].role == nullptr || messages[i].role[0] == '\0') {
                return fail("chat message role is empty");
            }
            if (messages[i].content == nullptr) {
                return fail("chat message content is null");
            }
            chat_messages.push_back({messages[i].role, messages[i].content});
        }

        const common_chat_params chat_params = format_chat_prompt_with_template(
            session->session->chat_templates.get(),
            chat_messages,
            params->add_generation_prompt != 0,
            chat_template_kwargs,
            native_reasoning_format);

        const std::vector<std::string> all_stops = merge_stop_strings(
            request_stops,
            chat_params.additional_stops);
        uint32_t finish_reason = MS_CHAT_FINISH_REASON_LENGTH;
        uint32_t actual_seed = LLAMA_DEFAULT_SEED;
        ms_baseline_benchmark benchmark = {};
        const int32_t result = run_session_generate_stream(
            *session->session,
            chat_params.prompt.c_str(),
            *params,
            all_stops,
            chat_params,
            native_reasoning_format,
            stream_callback,
            stream_user_data,
            &benchmark,
            finish_reason,
            actual_seed);
        if (result != 0) {
            return result;
        }

        benchmark.copied_tensor_count = session->session->copied_tensors;
        benchmark.converted_tensor_count = session->session->converted_tensors;
        benchmark.converted_bytes_before = session->session->converted_bytes_before;
        benchmark.converted_bytes_after = session->session->converted_bytes_after;
        benchmark.requested_target_count = session->session->requested_target_count;
        benchmark.verified_target_count = session->session->verified_target_count;
        out_result->benchmark = benchmark;
        out_result->prompt_tokens = benchmark.prompt_tokens;
        out_result->completion_tokens = benchmark.generated_tokens;
        out_result->finish_reason = finish_reason;
        out_result->actual_seed = actual_seed;
        session->completion_count += 1;
        return 0;
    } catch (const std::exception & err) {
        return fail(err.what());
    } catch (...) {
        return fail("unknown native recipe chat session streaming error");
    }
}

int32_t ms_runtime_generate_recipe_chat_session_multimodal_stream(
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
    ms_chat_generation_result * out_result) {
    clear_error();

    if (session == nullptr || session->session == nullptr) {
        return fail("recipe chat session is null");
    }
    if (messages == nullptr || message_count == 0) {
        return fail("chat message set is empty");
    }
    if (image_data_urls == nullptr || image_count == 0) {
        return fail("multimodal image input is empty");
    }
    if (params == nullptr) {
        return fail("chat generation params are null");
    }
    if (stop_strings == nullptr && stop_count > 0) {
        return fail("chat stop string pointer is null");
    }
    if (stream_callback == nullptr) {
        return fail("chat stream callback is null");
    }
    if (out_result == nullptr) {
        return fail("chat generation result output pointer is null");
    }

    try {
        std::vector<std::string> request_stops = collect_stop_strings(stop_strings, stop_count);
        const std::map<std::string, std::string> chat_template_kwargs =
            chat_template_kwargs_from_json(chat_template_kwargs_json);
        const common_reasoning_format native_reasoning_format =
            reasoning_format_from_request(reasoning_format);
        std::vector<std::pair<std::string, std::string>> chat_messages;
        chat_messages.reserve(static_cast<size_t>(message_count));
        for (uint64_t i = 0; i < message_count; ++i) {
            if (messages[i].role == nullptr || messages[i].role[0] == '\0') {
                return fail("chat message role is empty");
            }
            if (messages[i].content == nullptr) {
                return fail("chat message content is null");
            }
            chat_messages.push_back({messages[i].role, messages[i].content});
        }

        const common_chat_params chat_params = format_chat_prompt_with_template(
            session->session->chat_templates.get(),
            chat_messages,
            params->add_generation_prompt != 0,
            chat_template_kwargs,
            native_reasoning_format);
        const std::vector<std::string> all_stops = merge_stop_strings(
            request_stops,
            chat_params.additional_stops);

        uint32_t finish_reason = MS_CHAT_FINISH_REASON_LENGTH;
        uint32_t actual_seed = LLAMA_DEFAULT_SEED;
        ms_baseline_benchmark benchmark = {};
        const int32_t result = run_session_generate_multimodal_stream(
            *session->session,
            chat_params.prompt.c_str(),
            image_data_urls,
            image_count,
            *params,
            all_stops,
            chat_params,
            native_reasoning_format,
            stream_callback,
            stream_user_data,
            &benchmark,
            finish_reason,
            actual_seed);
        if (result != 0) {
            return result;
        }

        benchmark.copied_tensor_count = session->session->copied_tensors;
        benchmark.converted_tensor_count = session->session->converted_tensors;
        benchmark.converted_bytes_before = session->session->converted_bytes_before;
        benchmark.converted_bytes_after = session->session->converted_bytes_after;
        benchmark.requested_target_count = session->session->requested_target_count;
        benchmark.verified_target_count = session->session->verified_target_count;
        out_result->benchmark = benchmark;
        out_result->prompt_tokens = benchmark.prompt_tokens;
        out_result->completion_tokens = benchmark.generated_tokens;
        out_result->finish_reason = finish_reason;
        out_result->actual_seed = actual_seed;
        session->completion_count += 1;
        return 0;
    } catch (const std::exception & err) {
        return fail(err.what());
    } catch (...) {
        return fail("unknown native multimodal chat session streaming error");
    }
}

int32_t ms_runtime_get_recipe_chat_session_counters(
    const ms_runtime_chat_session * session,
    ms_runtime_chat_session_counters * out_counters) {
    clear_error();

    if (session == nullptr || session->session == nullptr) {
        return fail("recipe chat session is null");
    }

    if (out_counters == nullptr) {
        return fail("chat session counter output pointer is null");
    }

    out_counters->model_load_count = session->model_load_count;
    out_counters->context_reset_count = session->session->context_reset_count;
    out_counters->completion_count = session->completion_count;
    out_counters->copied_tensor_count = session->session->copied_tensors;
    out_counters->converted_tensor_count = session->session->converted_tensors;
    out_counters->converted_bytes_before = session->session->converted_bytes_before;
    out_counters->converted_bytes_after = session->session->converted_bytes_after;
    out_counters->requested_target_count = session->session->requested_target_count;
    out_counters->verified_target_count = session->session->verified_target_count;
    return 0;
}

int32_t ms_runtime_eval_recipe(
    const char * path,
    const ms_recipe_tensor_target * targets,
    uint64_t target_count,
    const char * const * eval_texts,
    uint64_t eval_text_count,
    uint32_t max_eval_tokens,
    const char * prompt,
    uint32_t max_tokens,
    ms_baseline_benchmark * out_benchmark,
    ms_recipe_eval_result * out_eval) {
    clear_error();

    if (path == nullptr || path[0] == '\0') {
        return fail("GGUF path is empty");
    }

    if (targets == nullptr && target_count > 0) {
        return fail("recipe target pointer is null");
    }

    if (eval_texts == nullptr || eval_text_count == 0) {
        return fail("eval text set is empty");
    }

    if (prompt == nullptr || prompt[0] == '\0') {
        return fail("benchmark prompt is empty");
    }

    if (out_benchmark == nullptr || out_eval == nullptr) {
        return fail("eval output pointer is null");
    }

    try {
        ensure_backend_initialized();

        std::unordered_map<std::string, ggml_type> target_types;
        if (!build_recipe_targets(targets, target_count, target_types)) {
            return -1;
        }

        PerplexityScore baseline_score = {};
        ms_baseline_benchmark baseline_benchmark = {};
        {
            std::unique_ptr<ModelSession> baseline_session = open_baseline_session(
                path,
                session_context_tokens(max_eval_tokens));
            baseline_score = score_session_perplexity(
                *baseline_session,
                eval_texts,
                eval_text_count,
                max_eval_tokens);
            const int32_t baseline_result = run_session_benchmark(
                *baseline_session,
                prompt,
                max_tokens,
                &baseline_benchmark);
            if (baseline_result != 0) {
                return baseline_result;
            }
        }

        std::unique_ptr<ModelSession> recipe_session = open_recipe_session(
            path,
            std::move(target_types),
            session_context_tokens(max_eval_tokens));
        if (recipe_session == nullptr) {
            return -1;
        }

        PerplexityScore recipe_score = score_session_perplexity(
            *recipe_session,
            eval_texts,
            eval_text_count,
            max_eval_tokens);

        const int32_t benchmark_result = run_session_benchmark(
            *recipe_session,
            prompt,
            max_tokens,
            out_benchmark);
        if (benchmark_result != 0) {
            return benchmark_result;
        }

        out_benchmark->copied_tensor_count = recipe_session->copied_tensors;
        out_benchmark->converted_tensor_count = recipe_session->converted_tensors;
        out_benchmark->converted_bytes_before = recipe_session->converted_bytes_before;
        out_benchmark->converted_bytes_after = recipe_session->converted_bytes_after;
        out_benchmark->requested_target_count = recipe_session->requested_target_count;
        out_benchmark->verified_target_count = recipe_session->verified_target_count;

        out_eval->baseline_load_ms = baseline_benchmark.load_ms;
        out_eval->baseline_prompt_eval_ms = baseline_benchmark.prompt_eval_ms;
        out_eval->baseline_generation_ms = baseline_benchmark.generation_ms;
        out_eval->baseline_prompt_eval_tps = baseline_benchmark.prompt_eval_tps;
        out_eval->baseline_token_gen_tps = baseline_benchmark.token_gen_tps;
        out_eval->baseline_ttft_ms = baseline_benchmark.ttft_ms;
        out_eval->baseline_runtime_elapsed_ms = benchmark_runtime_elapsed_ms(baseline_benchmark);
        out_eval->baseline_nll = baseline_score.total_nll / static_cast<double>(baseline_score.token_count);
        out_eval->baseline_ppl = baseline_score.ppl;
        out_eval->baseline_ppl_uncertainty = baseline_score.ppl_uncertainty;
        out_eval->baseline_eval_ms = baseline_score.eval_ms;
        out_eval->baseline_vram_peak_mb = baseline_benchmark.vram_peak_mb;
        out_eval->baseline_vram_allocated_mb = baseline_benchmark.vram_allocated_mb;
        out_eval->recipe_nll = recipe_score.total_nll / static_cast<double>(recipe_score.token_count);
        out_eval->recipe_ppl = recipe_score.ppl;
        out_eval->recipe_ppl_uncertainty = recipe_score.ppl_uncertainty;
        out_eval->recipe_eval_ms = recipe_score.eval_ms;
        out_eval->recipe_vram_peak_mb = recipe_score.vram_peak_mb;
        out_eval->recipe_vram_allocated_mb = recipe_score.vram_allocated_mb;
        out_eval->ppl_delta = out_eval->recipe_ppl - out_eval->baseline_ppl;
        out_eval->ppl_delta_percent = out_eval->baseline_ppl > 0.0
            ? (out_eval->ppl_delta / out_eval->baseline_ppl) * 100.0
            : 0.0;
        out_eval->eval_token_count = recipe_score.token_count;
        out_eval->eval_sample_count = recipe_score.sample_count;
        out_eval->skipped_sample_count = baseline_score.skipped_count + recipe_score.skipped_count;
        return 0;
    } catch (const std::exception & err) {
        return fail(err.what());
    } catch (...) {
        return fail("unknown native recipe eval error");
    }
}

int32_t ms_runtime_eval_recipe_single(
    const char * path,
    const ms_recipe_tensor_target * targets,
    uint64_t target_count,
    const char * const * eval_texts,
    uint64_t eval_text_count,
    uint32_t max_eval_tokens,
    const char * prompt,
    uint32_t max_tokens,
    ms_baseline_benchmark * out_benchmark,
    ms_recipe_eval_result * out_eval) {
    clear_error();

    if (path == nullptr || path[0] == '\0') {
        return fail("GGUF path is empty");
    }

    if (targets == nullptr && target_count > 0) {
        return fail("recipe target pointer is null");
    }

    if (eval_texts == nullptr || eval_text_count == 0) {
        return fail("eval text set is empty");
    }

    if (prompt == nullptr || prompt[0] == '\0') {
        return fail("benchmark prompt is empty");
    }

    if (out_benchmark == nullptr || out_eval == nullptr) {
        return fail("eval output pointer is null");
    }

    try {
        ensure_backend_initialized();

        std::unordered_map<std::string, ggml_type> target_types;
        if (!build_recipe_targets(targets, target_count, target_types)) {
            return -1;
        }

        std::unique_ptr<ModelSession> recipe_session = open_recipe_session(
            path,
            std::move(target_types),
            session_context_tokens(max_eval_tokens));
        if (recipe_session == nullptr) {
            return -1;
        }

        PerplexityScore recipe_score = score_session_perplexity(
            *recipe_session,
            eval_texts,
            eval_text_count,
            max_eval_tokens);

        const int32_t benchmark_result = run_session_benchmark(
            *recipe_session,
            prompt,
            max_tokens,
            out_benchmark);
        if (benchmark_result != 0) {
            return benchmark_result;
        }

        out_benchmark->copied_tensor_count = recipe_session->copied_tensors;
        out_benchmark->converted_tensor_count = recipe_session->converted_tensors;
        out_benchmark->converted_bytes_before = recipe_session->converted_bytes_before;
        out_benchmark->converted_bytes_after = recipe_session->converted_bytes_after;
        out_benchmark->requested_target_count = recipe_session->requested_target_count;
        out_benchmark->verified_target_count = recipe_session->verified_target_count;

        *out_eval = {};
        out_eval->recipe_nll = recipe_score.total_nll / static_cast<double>(recipe_score.token_count);
        out_eval->recipe_ppl = recipe_score.ppl;
        out_eval->recipe_ppl_uncertainty = recipe_score.ppl_uncertainty;
        out_eval->recipe_eval_ms = recipe_score.eval_ms;
        out_eval->recipe_vram_peak_mb = recipe_score.vram_peak_mb;
        out_eval->recipe_vram_allocated_mb = recipe_score.vram_allocated_mb;
        out_eval->eval_token_count = recipe_score.token_count;
        out_eval->eval_sample_count = recipe_score.sample_count;
        out_eval->skipped_sample_count = recipe_score.skipped_count;
        return 0;
    } catch (const std::exception & err) {
        return fail(err.what());
    } catch (...) {
        return fail("unknown native single recipe eval error");
    }
}

int32_t ms_runtime_eval_recipe_standard(
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
    uint64_t * out_sample_audit_count) {
    clear_error();

    if (path == nullptr || path[0] == '\0') {
        return fail("GGUF path is empty");
    }

    if (targets == nullptr && target_count > 0) {
        return fail("recipe target pointer is null");
    }

    if (eval_texts == nullptr || eval_text_count == 0) {
        return fail("eval text set is empty");
    }

    if (standard_samples == nullptr && standard_sample_count > 0) {
        return fail("standard eval sample pointer is null");
    }

    if (prompt == nullptr || prompt[0] == '\0') {
        return fail("benchmark prompt is empty");
    }

    if (out_benchmark == nullptr || out_eval == nullptr || out_task_result_count == nullptr || out_sample_audit_count == nullptr) {
        return fail("standard eval output pointer is null");
    }

    try {
        ensure_backend_initialized();

        std::unordered_map<std::string, ggml_type> target_types;
        if (!build_recipe_targets(targets, target_count, target_types)) {
            return -1;
        }

        PerplexityScore baseline_score = {};
        ms_baseline_benchmark baseline_benchmark = {};
        std::vector<StandardEvalSampleScore> baseline_standard_scores;
        {
            std::unique_ptr<ModelSession> baseline_session = open_baseline_session(
                path,
                session_context_tokens(max_eval_tokens));
            baseline_score = score_session_perplexity(
                *baseline_session,
                eval_texts,
                eval_text_count,
                max_eval_tokens);
            baseline_standard_scores = score_standard_eval_samples(
                *baseline_session,
                standard_samples,
                standard_sample_count);
            const int32_t baseline_result = run_session_benchmark(
                *baseline_session,
                prompt,
                max_tokens,
                &baseline_benchmark);
            if (baseline_result != 0) {
                return baseline_result;
            }
        }

        std::unique_ptr<ModelSession> recipe_session = open_recipe_session(
            path,
            std::move(target_types),
            session_context_tokens(max_eval_tokens));
        if (recipe_session == nullptr) {
            return -1;
        }

        PerplexityScore recipe_score = score_session_perplexity(
            *recipe_session,
            eval_texts,
            eval_text_count,
            max_eval_tokens);
        std::vector<StandardEvalSampleScore> recipe_standard_scores = score_standard_eval_samples(
            *recipe_session,
            standard_samples,
            standard_sample_count);

        const int32_t benchmark_result = run_session_benchmark(
            *recipe_session,
            prompt,
            max_tokens,
            out_benchmark);
        if (benchmark_result != 0) {
            return benchmark_result;
        }

        out_benchmark->copied_tensor_count = recipe_session->copied_tensors;
        out_benchmark->converted_tensor_count = recipe_session->converted_tensors;
        out_benchmark->converted_bytes_before = recipe_session->converted_bytes_before;
        out_benchmark->converted_bytes_after = recipe_session->converted_bytes_after;
        out_benchmark->requested_target_count = recipe_session->requested_target_count;
        out_benchmark->verified_target_count = recipe_session->verified_target_count;

        out_eval->baseline_load_ms = baseline_benchmark.load_ms;
        out_eval->baseline_prompt_eval_ms = baseline_benchmark.prompt_eval_ms;
        out_eval->baseline_generation_ms = baseline_benchmark.generation_ms;
        out_eval->baseline_prompt_eval_tps = baseline_benchmark.prompt_eval_tps;
        out_eval->baseline_token_gen_tps = baseline_benchmark.token_gen_tps;
        out_eval->baseline_ttft_ms = baseline_benchmark.ttft_ms;
        out_eval->baseline_runtime_elapsed_ms = benchmark_runtime_elapsed_ms(baseline_benchmark);
        out_eval->baseline_nll = baseline_score.total_nll / static_cast<double>(baseline_score.token_count);
        out_eval->baseline_ppl = baseline_score.ppl;
        out_eval->baseline_ppl_uncertainty = baseline_score.ppl_uncertainty;
        out_eval->baseline_eval_ms = baseline_score.eval_ms;
        out_eval->baseline_vram_peak_mb = baseline_benchmark.vram_peak_mb;
        out_eval->baseline_vram_allocated_mb = baseline_benchmark.vram_allocated_mb;
        out_eval->recipe_nll = recipe_score.total_nll / static_cast<double>(recipe_score.token_count);
        out_eval->recipe_ppl = recipe_score.ppl;
        out_eval->recipe_ppl_uncertainty = recipe_score.ppl_uncertainty;
        out_eval->recipe_eval_ms = recipe_score.eval_ms;
        out_eval->recipe_vram_peak_mb = recipe_score.vram_peak_mb;
        out_eval->recipe_vram_allocated_mb = recipe_score.vram_allocated_mb;
        out_eval->ppl_delta = out_eval->recipe_ppl - out_eval->baseline_ppl;
        out_eval->ppl_delta_percent = out_eval->baseline_ppl > 0.0
            ? (out_eval->ppl_delta / out_eval->baseline_ppl) * 100.0
            : 0.0;
        out_eval->eval_token_count = recipe_score.token_count;
        out_eval->eval_sample_count = recipe_score.sample_count;
        out_eval->skipped_sample_count = baseline_score.skipped_count + recipe_score.skipped_count;

        if (!write_standard_eval_results(
                &baseline_standard_scores,
                recipe_standard_scores,
                out_task_results,
                task_result_capacity,
                out_task_result_count)) {
            return -1;
        }
        if (!write_standard_eval_sample_audits(
                &baseline_standard_scores,
                recipe_standard_scores,
                out_sample_audits,
                sample_audit_capacity,
                out_sample_audit_count)) {
            return -1;
        }
        return 0;
    } catch (const std::exception & err) {
        return fail(err.what());
    } catch (...) {
        return fail("unknown native standard recipe eval error");
    }
}

int32_t ms_runtime_eval_recipe_standard_single(
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
    uint64_t * out_sample_audit_count) {
    clear_error();

    if (path == nullptr || path[0] == '\0') {
        return fail("GGUF path is empty");
    }

    if (targets == nullptr && target_count > 0) {
        return fail("recipe target pointer is null");
    }

    if (eval_texts == nullptr || eval_text_count == 0) {
        return fail("eval text set is empty");
    }

    if (standard_samples == nullptr && standard_sample_count > 0) {
        return fail("standard eval sample pointer is null");
    }

    if (prompt == nullptr || prompt[0] == '\0') {
        return fail("benchmark prompt is empty");
    }

    if (out_benchmark == nullptr || out_eval == nullptr || out_task_result_count == nullptr || out_sample_audit_count == nullptr) {
        return fail("standard eval output pointer is null");
    }

    try {
        ensure_backend_initialized();

        std::unordered_map<std::string, ggml_type> target_types;
        if (!build_recipe_targets(targets, target_count, target_types)) {
            return -1;
        }

        std::unique_ptr<ModelSession> recipe_session = open_recipe_session(
            path,
            std::move(target_types),
            session_context_tokens(max_eval_tokens));
        if (recipe_session == nullptr) {
            return -1;
        }

        PerplexityScore recipe_score = score_session_perplexity(
            *recipe_session,
            eval_texts,
            eval_text_count,
            max_eval_tokens);
        std::vector<StandardEvalSampleScore> recipe_standard_scores = score_standard_eval_samples(
            *recipe_session,
            standard_samples,
            standard_sample_count);

        const int32_t benchmark_result = run_session_benchmark(
            *recipe_session,
            prompt,
            max_tokens,
            out_benchmark);
        if (benchmark_result != 0) {
            return benchmark_result;
        }

        out_benchmark->copied_tensor_count = recipe_session->copied_tensors;
        out_benchmark->converted_tensor_count = recipe_session->converted_tensors;
        out_benchmark->converted_bytes_before = recipe_session->converted_bytes_before;
        out_benchmark->converted_bytes_after = recipe_session->converted_bytes_after;
        out_benchmark->requested_target_count = recipe_session->requested_target_count;
        out_benchmark->verified_target_count = recipe_session->verified_target_count;

        *out_eval = {};
        out_eval->recipe_nll = recipe_score.total_nll / static_cast<double>(recipe_score.token_count);
        out_eval->recipe_ppl = recipe_score.ppl;
        out_eval->recipe_ppl_uncertainty = recipe_score.ppl_uncertainty;
        out_eval->recipe_eval_ms = recipe_score.eval_ms;
        out_eval->recipe_vram_peak_mb = recipe_score.vram_peak_mb;
        out_eval->recipe_vram_allocated_mb = recipe_score.vram_allocated_mb;
        out_eval->eval_token_count = recipe_score.token_count;
        out_eval->eval_sample_count = recipe_score.sample_count;
        out_eval->skipped_sample_count = recipe_score.skipped_count;

        if (!write_standard_eval_results(
                nullptr,
                recipe_standard_scores,
                out_task_results,
                task_result_capacity,
                out_task_result_count)) {
            return -1;
        }
        if (!write_standard_eval_sample_audits(
                nullptr,
                recipe_standard_scores,
                out_sample_audits,
                sample_audit_capacity,
                out_sample_audit_count)) {
            return -1;
        }
        return 0;
    } catch (const std::exception & err) {
        return fail(err.what());
    } catch (...) {
        return fail("unknown native single standard recipe eval error");
    }
}

} // extern "C"
