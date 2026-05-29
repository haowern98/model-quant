#include "model_surgery_runtime.h"

#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"

#include <algorithm>
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
#include <unordered_map>
#include <vector>

#if defined(MODEL_SURGERY_RUNTIME_CUDA_PROFILING)
#include <cuda_runtime.h>
#endif

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

bool is_runtime_quantizable_tensor(std::string_view name) {
    return name.find("bias") == std::string_view::npos
        && name.find("norm") == std::string_view::npos
        && name.find("rope") == std::string_view::npos
        && name.find("scale") == std::string_view::npos;
}

bool supports_recipe_conversion(std::string_view name, ggml_type current_type, ggml_type target_type) {
    if (target_type != GGML_TYPE_Q8_0) {
        return false;
    }

    if (!is_runtime_quantizable_tensor(name)) {
        return false;
    }

    return current_type == GGML_TYPE_F32
        || current_type == GGML_TYPE_F16
        || current_type == GGML_TYPE_BF16;
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
            + ". Phase 2 supports only F32/F16/BF16->Q8_0 conversions.");
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
    size_t data_offset = 0;
    uint64_t copied_tensors = 0;
    uint64_t converted_tensors = 0;
    uint64_t converted_bytes_before = 0;
    uint64_t converted_bytes_after = 0;
};

struct PerplexityScore {
    double total_nll = 0.0;
    double ppl = 0.0;
    double eval_ms = 0.0;
    double vram_peak_mb = 0.0;
    double vram_allocated_mb = 0.0;
    uint64_t token_count = 0;
    uint64_t sample_count = 0;
    uint64_t skipped_count = 0;
};

struct StandardEvalSampleScore {
    std::string task;
    uint32_t gold_index = 0;
    uint32_t prediction_index = 0;
    bool correct = false;
    double margin = 0.0;
    double correct_nll = 0.0;
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
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx;
    VramTracker vram;
    double load_ms = 0.0;
    uint64_t copied_tensors = 0;
    uint64_t converted_tensors = 0;
    uint64_t converted_bytes_before = 0;
    uint64_t converted_bytes_after = 0;

    ModelSession(llama_model * loaded_model, double load_time_ms, const VramTracker & vram_tracker)
        : model(loaded_model, llama_model_free),
          ctx(nullptr, llama_free),
          vram(vram_tracker),
          load_ms(load_time_ms) {
    }
};

uint32_t session_context_tokens(uint32_t max_eval_tokens) {
    const uint32_t eval_limit = std::max<uint32_t>(2, max_eval_tokens == 0 ? 128 : max_eval_tokens);
    return std::max<uint32_t>(512, eval_limit + 8);
}

void open_session_context(ModelSession & session, uint32_t context_tokens) {
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
    session.vram.sample();
}

void reset_session_context(ModelSession & session) {
    if (session.ctx == nullptr) {
        throw std::runtime_error("model session context is not initialized");
    }
    llama_memory_clear(llama_get_memory(session.ctx.get()), true);
}

std::unique_ptr<ModelSession> open_baseline_session(const char * path, uint32_t context_tokens) {
    VramTracker vram_tracker = {};
    vram_tracker.reset();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = -1;
    model_params.use_mmap = true;

    const auto load_start = std::chrono::steady_clock::now();
    llama_model * model = llama_model_load_from_file(path, model_params);
    const auto load_end = std::chrono::steady_clock::now();
    if (model == nullptr) {
        throw std::runtime_error(std::string("failed to load model: ") + path);
    }
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

std::unique_ptr<ModelSession> open_recipe_session(
    const char * path,
    std::unordered_map<std::string, ggml_type> target_types,
    uint32_t context_tokens) {
    gguf_init_params gguf_params = {};
    gguf_params.no_alloc = true;
    gguf_params.ctx = nullptr;

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

    if (!validate_recipe_for_user_model(source_metadata.get(), target_types)) {
        return nullptr;
    }
    apply_recipe_tensor_types(model_metadata.get(), target_types);

    UserCopyTensorReader reader = {};
    reader.source_metadata = source_metadata.get();
    reader.target_types = std::move(target_types);
    reader.data_offset = gguf_get_data_offset(source_metadata.get());
    reader.file.open(path, std::ios::binary);
    if (!reader.file.is_open()) {
        throw std::runtime_error(std::string("failed to open GGUF tensor data: ") + path);
    }

    VramTracker vram_tracker = {};
    vram_tracker.reset();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = -1;
    model_params.use_mmap = false;

    const auto load_start = std::chrono::steady_clock::now();
    llama_model * model = llama_model_init_from_user(
        model_metadata.get(),
        copy_user_tensor_data,
        &reader,
        model_params);
    const auto load_end = std::chrono::steady_clock::now();
    if (model == nullptr) {
        throw std::runtime_error(std::string("failed to load recipe model: ") + path);
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

void convert_source_tensor_to_q8_0(
    ggml_tensor * tensor,
    UserCopyTensorReader * reader,
    const char * name,
    int64_t tensor_id,
    ggml_type current_type) {
    const int64_t n_per_row = tensor->ne[0];
    if (n_per_row <= 0) {
        throw std::runtime_error(std::string("cannot quantize tensor with empty row: ") + name);
    }

    const int64_t q8_block = ggml_blck_size(GGML_TYPE_Q8_0);
    if (n_per_row % q8_block != 0) {
        throw std::runtime_error(
            std::string("cannot quantize tensor to Q8_0 because row size is not divisible by ")
            + std::to_string(q8_block) + ": " + name);
    }

    const int64_t element_count = ggml_nelements(tensor);
    if (element_count % n_per_row != 0) {
        throw std::runtime_error(std::string("cannot quantize tensor with irregular row layout: ") + name);
    }

    const int64_t nrows = element_count / n_per_row;
    const size_t source_row_size = ggml_row_size(current_type, n_per_row);
    const size_t target_row_size = ggml_row_size(GGML_TYPE_Q8_0, n_per_row);
    const size_t source_size = gguf_get_tensor_size(reader->source_metadata, tensor_id);
    const size_t target_size = ggml_nbytes(tensor);

    if (source_size != source_row_size * static_cast<size_t>(nrows)) {
        throw std::runtime_error(std::string("source tensor size does not match row layout for: ") + name);
    }
    if (target_size != target_row_size * static_cast<size_t>(nrows)) {
        throw std::runtime_error(std::string("target tensor size does not match Q8_0 row layout for: ") + name);
    }

    reader->source_row_buffer.resize(source_row_size);
    reader->f32_row_buffer.resize(static_cast<size_t>(n_per_row));
    reader->quantized_row_buffer.resize(target_row_size);

    const size_t source_offset = reader->data_offset + gguf_get_tensor_offset(reader->source_metadata, tensor_id);
    for (int64_t row = 0; row < nrows; ++row) {
        read_exact_at(
            reader,
            source_offset + static_cast<size_t>(row) * source_row_size,
            reader->source_row_buffer.data(),
            source_row_size,
            name);

        if (current_type == GGML_TYPE_F32) {
            std::memcpy(
                reader->f32_row_buffer.data(),
                reader->source_row_buffer.data(),
                source_row_size);
        } else if (current_type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row(
                reinterpret_cast<const ggml_fp16_t *>(reader->source_row_buffer.data()),
                reader->f32_row_buffer.data(),
                n_per_row);
        } else if (current_type == GGML_TYPE_BF16) {
            ggml_bf16_to_fp32_row(
                reinterpret_cast<const ggml_bf16_t *>(reader->source_row_buffer.data()),
                reader->f32_row_buffer.data(),
                n_per_row);
        } else {
            throw std::runtime_error(
                std::string("unsupported source quant for Q8_0 conversion: ")
                + name + " " + display_quant_type(current_type));
        }

        const size_t written = ggml_quantize_chunk(
            GGML_TYPE_Q8_0,
            reader->f32_row_buffer.data(),
            reader->quantized_row_buffer.data(),
            0,
            1,
            n_per_row,
            nullptr);
        if (written != target_row_size) {
            throw std::runtime_error(std::string("Q8_0 quantized row size mismatch for: ") + name);
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
        if (supports_recipe_conversion(name, current_type, target_type) && target_type == GGML_TYPE_Q8_0) {
            convert_source_tensor_to_q8_0(tensor, reader, name, tensor_id, current_type);
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
    const char * text,
    uint32_t max_eval_tokens) {
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
    if (max_eval_tokens > 0 && tokens.size() > max_eval_tokens) {
        tokens.resize(static_cast<size_t>(max_eval_tokens));
    }
    return tokens;
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

double score_continuation_nll(
    ModelSession & session,
    const char * prompt,
    const char * continuation,
    uint64_t * out_token_count) {
    if (prompt == nullptr || prompt[0] == '\0') {
        throw std::runtime_error("standard eval sample prompt is empty");
    }
    if (continuation == nullptr || continuation[0] == '\0') {
        throw std::runtime_error("standard eval sample choice is empty");
    }

    const llama_vocab * vocab = llama_model_get_vocab(session.model.get());
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    if (n_vocab <= 0) {
        throw std::runtime_error("model vocabulary is empty");
    }

    const int32_t prompt_len = static_cast<int32_t>(std::strlen(prompt));
    int32_t prompt_token_count = llama_tokenize(vocab, prompt, prompt_len, nullptr, 0, true, true);
    if (prompt_token_count == INT32_MIN) {
        throw std::runtime_error("standard eval prompt tokenization overflowed");
    }
    if (prompt_token_count < 0) {
        prompt_token_count = -prompt_token_count;
    }
    if (prompt_token_count <= 0) {
        throw std::runtime_error("standard eval prompt produced no tokens");
    }

    std::vector<llama_token> prompt_tokens(static_cast<size_t>(prompt_token_count));
    const int32_t actual_prompt_tokens = llama_tokenize(
        vocab,
        prompt,
        prompt_len,
        prompt_tokens.data(),
        prompt_token_count,
        true,
        true);
    if (actual_prompt_tokens <= 0) {
        throw std::runtime_error("failed to tokenize standard eval prompt");
    }
    prompt_tokens.resize(static_cast<size_t>(actual_prompt_tokens));

    const int32_t continuation_len = static_cast<int32_t>(std::strlen(continuation));
    int32_t continuation_token_count = llama_tokenize(
        vocab,
        continuation,
        continuation_len,
        nullptr,
        0,
        false,
        true);
    if (continuation_token_count == INT32_MIN) {
        throw std::runtime_error("standard eval choice tokenization overflowed");
    }
    if (continuation_token_count < 0) {
        continuation_token_count = -continuation_token_count;
    }
    if (continuation_token_count <= 0) {
        throw std::runtime_error("standard eval choice produced no tokens");
    }

    std::vector<llama_token> continuation_tokens(static_cast<size_t>(continuation_token_count));
    const int32_t actual_continuation_tokens = llama_tokenize(
        vocab,
        continuation,
        continuation_len,
        continuation_tokens.data(),
        continuation_token_count,
        false,
        true);
    if (actual_continuation_tokens <= 0) {
        throw std::runtime_error("failed to tokenize standard eval choice");
    }
    continuation_tokens.resize(static_cast<size_t>(actual_continuation_tokens));

    reset_session_context(session);
    const int prompt_res = llama_decode(
        session.ctx.get(),
        llama_batch_get_one(prompt_tokens.data(), static_cast<int32_t>(prompt_tokens.size())));
    if (prompt_res != 0) {
        throw std::runtime_error("failed to decode standard eval prompt");
    }
    llama_synchronize(session.ctx.get());

    double nll = 0.0;
    for (llama_token token : continuation_tokens) {
        const float * logits = llama_get_logits_ith(session.ctx.get(), -1);
        nll += token_nll_from_logits(logits, n_vocab, token);

        llama_batch batch = llama_batch_get_one(&token, 1);
        const int decode_result = llama_decode(session.ctx.get(), batch);
        if (decode_result != 0) {
            throw std::runtime_error("failed to decode standard eval choice token");
        }
        llama_synchronize(session.ctx.get());
    }

    if (out_token_count != nullptr) {
        *out_token_count = static_cast<uint64_t>(continuation_tokens.size());
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
        if (sample.gold_index >= sample.choice_count) {
            throw std::runtime_error(std::string("standard eval gold index is outside choices for task: ") + sample.task);
        }

        std::vector<double> choice_scores(static_cast<size_t>(sample.choice_count));
        std::vector<double> choice_nlls(static_cast<size_t>(sample.choice_count));
        for (uint64_t choice_index = 0; choice_index < sample.choice_count; ++choice_index) {
            uint64_t token_count = 0;
            const double nll = score_continuation_nll(
                session,
                sample.prompt,
                sample.choices[choice_index],
                &token_count);
            const double denominator = sample.normalize_by_choice_length != 0 && token_count > 0
                ? static_cast<double>(token_count)
                : 1.0;
            choice_nlls[static_cast<size_t>(choice_index)] = nll;
            choice_scores[static_cast<size_t>(choice_index)] = -nll / denominator;
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
        score.gold_index = sample.gold_index;
        score.prediction_index = best_index;
        score.correct = best_index == sample.gold_index;
        score.margin = choice_scores[best_index] - choice_scores[second_index];
        score.correct_nll = choice_nlls[static_cast<size_t>(sample.gold_index)];
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

PerplexityScore score_session_perplexity(
    ModelSession & session,
    const char * const * eval_texts,
    uint64_t eval_text_count,
    uint32_t max_eval_tokens) {
    PerplexityScore score = {};
    if (eval_texts == nullptr || eval_text_count == 0) {
        throw std::runtime_error("eval text set is empty");
    }

    const llama_vocab * vocab = llama_model_get_vocab(session.model.get());
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    if (n_vocab <= 0) {
        throw std::runtime_error("model vocabulary is empty");
    }

    const uint32_t eval_limit = std::max<uint32_t>(2, max_eval_tokens == 0 ? 128 : max_eval_tokens);

    const auto eval_start = std::chrono::steady_clock::now();
    for (uint64_t text_index = 0; text_index < eval_text_count; ++text_index) {
        const char * text = eval_texts[text_index];
        if (text == nullptr || text[0] == '\0') {
            score.skipped_count += 1;
            continue;
        }

        std::vector<llama_token> tokens = tokenize_text(vocab, text, eval_limit);
        if (tokens.size() < 2) {
            score.skipped_count += 1;
            continue;
        }

        reset_session_context(session);
        for (size_t token_index = 0; token_index + 1 < tokens.size(); ++token_index) {
            llama_token token = tokens[token_index];
            llama_batch batch = llama_batch_get_one(&token, 1);
            const int decode_result = llama_decode(session.ctx.get(), batch);
            if (decode_result != 0) {
                throw std::runtime_error("failed to decode eval token");
            }
            llama_synchronize(session.ctx.get());

            const float * logits = llama_get_logits_ith(session.ctx.get(), -1);
            score.total_nll += token_nll_from_logits(logits, n_vocab, tokens[token_index + 1]);
            score.token_count += 1;
        }

        score.sample_count += 1;
        session.vram.sample();
    }
    const auto eval_end = std::chrono::steady_clock::now();

    if (score.token_count == 0) {
        throw std::runtime_error("eval text set produced no scoreable tokens");
    }

    score.eval_ms = elapsed_ms(eval_start, eval_end);
    const double avg_nll = score.total_nll / static_cast<double>(score.token_count);
    score.ppl = std::exp(std::min(avg_nll, 700.0));
    score.vram_peak_mb = session.vram.peak_mb;
    score.vram_allocated_mb = session.vram.current_mb;
    return score;
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
    const int prompt_res = llama_decode(
        session.ctx.get(),
        llama_batch_get_one(prompt_tokens.data(), static_cast<int32_t>(prompt_tokens.size())));
    if (prompt_res != 0) {
        return fail("failed to decode prompt");
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
        }
        return result;
    } catch (const std::exception & err) {
        return fail(err.what());
    } catch (...) {
        return fail("unknown native recipe benchmark error");
    }
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

        out_eval->baseline_load_ms = baseline_benchmark.load_ms;
        out_eval->baseline_prompt_eval_ms = baseline_benchmark.prompt_eval_ms;
        out_eval->baseline_generation_ms = baseline_benchmark.generation_ms;
        out_eval->baseline_prompt_eval_tps = baseline_benchmark.prompt_eval_tps;
        out_eval->baseline_token_gen_tps = baseline_benchmark.token_gen_tps;
        out_eval->baseline_ttft_ms = baseline_benchmark.ttft_ms;
        out_eval->baseline_runtime_elapsed_ms = benchmark_runtime_elapsed_ms(baseline_benchmark);
        out_eval->baseline_nll = baseline_score.total_nll / static_cast<double>(baseline_score.token_count);
        out_eval->baseline_ppl = baseline_score.ppl;
        out_eval->baseline_eval_ms = baseline_score.eval_ms;
        out_eval->baseline_vram_peak_mb = baseline_benchmark.vram_peak_mb;
        out_eval->baseline_vram_allocated_mb = baseline_benchmark.vram_allocated_mb;
        out_eval->recipe_nll = recipe_score.total_nll / static_cast<double>(recipe_score.token_count);
        out_eval->recipe_ppl = recipe_score.ppl;
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

        *out_eval = {};
        out_eval->recipe_nll = recipe_score.total_nll / static_cast<double>(recipe_score.token_count);
        out_eval->recipe_ppl = recipe_score.ppl;
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
    uint64_t * out_task_result_count) {
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

    if (out_benchmark == nullptr || out_eval == nullptr || out_task_result_count == nullptr) {
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

        out_eval->baseline_load_ms = baseline_benchmark.load_ms;
        out_eval->baseline_prompt_eval_ms = baseline_benchmark.prompt_eval_ms;
        out_eval->baseline_generation_ms = baseline_benchmark.generation_ms;
        out_eval->baseline_prompt_eval_tps = baseline_benchmark.prompt_eval_tps;
        out_eval->baseline_token_gen_tps = baseline_benchmark.token_gen_tps;
        out_eval->baseline_ttft_ms = baseline_benchmark.ttft_ms;
        out_eval->baseline_runtime_elapsed_ms = benchmark_runtime_elapsed_ms(baseline_benchmark);
        out_eval->baseline_nll = baseline_score.total_nll / static_cast<double>(baseline_score.token_count);
        out_eval->baseline_ppl = baseline_score.ppl;
        out_eval->baseline_eval_ms = baseline_score.eval_ms;
        out_eval->baseline_vram_peak_mb = baseline_benchmark.vram_peak_mb;
        out_eval->baseline_vram_allocated_mb = baseline_benchmark.vram_allocated_mb;
        out_eval->recipe_nll = recipe_score.total_nll / static_cast<double>(recipe_score.token_count);
        out_eval->recipe_ppl = recipe_score.ppl;
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
    uint64_t * out_task_result_count) {
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

    if (out_benchmark == nullptr || out_eval == nullptr || out_task_result_count == nullptr) {
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

        *out_eval = {};
        out_eval->recipe_nll = recipe_score.total_nll / static_cast<double>(recipe_score.token_count);
        out_eval->recipe_ppl = recipe_score.ppl;
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
        return 0;
    } catch (const std::exception & err) {
        return fail(err.what());
    } catch (...) {
        return fail("unknown native single standard recipe eval error");
    }
}

} // extern "C"
