#include "../src/model_surgery_runtime.cpp"

#include "chat-peg-parser.h"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>

namespace {

void collect_runtime_log(const char * message, void * user_data) {
    auto * logs = static_cast<std::vector<std::string> *>(user_data);
    logs->emplace_back(message == nullptr ? "" : message);
}

void test_k_quant_recipe_targets_are_supported() {
    assert(supports_recipe_conversion(
        "layers.0.attn_q.weight",
        GGML_TYPE_F16,
        GGML_TYPE_Q4_K));
    assert(supports_recipe_conversion(
        "layers.0.attn_q.weight",
        GGML_TYPE_Q8_0,
        GGML_TYPE_Q4_K));
    assert(!supports_recipe_conversion(
        "layers.0.attn_norm.weight",
        GGML_TYPE_F16,
        GGML_TYPE_Q4_K));
    assert(!supports_recipe_conversion(
        "layers.0.attn_q.weight",
        GGML_TYPE_Q4_K,
        GGML_TYPE_Q8_0));
}

void test_legacy_quant_recipe_targets_are_supported() {
    ggml_type parsed = GGML_TYPE_COUNT;
    assert(parse_quant_type("Q5_1", parsed));
    assert(parsed == GGML_TYPE_Q5_1);
    assert(parse_quant_type("Q5_0", parsed));
    assert(parsed == GGML_TYPE_Q5_0);
    assert(parse_quant_type("Q4_1", parsed));
    assert(parsed == GGML_TYPE_Q4_1);
    assert(parse_quant_type("Q4_0", parsed));
    assert(parsed == GGML_TYPE_Q4_0);

    assert(supports_recipe_conversion(
        "layers.0.attn_q.weight",
        GGML_TYPE_F16,
        GGML_TYPE_Q4_0));
    assert(supports_recipe_conversion(
        "layers.0.attn_q.weight",
        GGML_TYPE_Q8_0,
        GGML_TYPE_Q4_1));
    assert(supports_recipe_conversion(
        "layers.0.attn_q.weight",
        GGML_TYPE_Q5_1,
        GGML_TYPE_Q5_0));
}

void test_recipe_conversion_rejects_cross_family_quant_targets() {
    assert(!supports_recipe_conversion(
        "layers.0.attn_q.weight",
        GGML_TYPE_Q5_0,
        GGML_TYPE_Q4_K));
    assert(!supports_recipe_conversion(
        "layers.0.attn_q.weight",
        GGML_TYPE_Q6_K,
        GGML_TYPE_Q4_0));
}

void test_quantized_source_rows_can_decode_to_f32() {
    constexpr int64_t n_per_row = 256;
    std::vector<float> source(n_per_row);
    for (int64_t i = 0; i < n_per_row; ++i) {
        source[static_cast<size_t>(i)] = static_cast<float>((i % 17) - 8);
    }

    std::vector<char> q8(ggml_row_size(GGML_TYPE_Q8_0, n_per_row));
    const size_t written = ggml_quantize_chunk(
        GGML_TYPE_Q8_0,
        source.data(),
        q8.data(),
        0,
        1,
        n_per_row,
        nullptr);
    assert(written == q8.size());

    std::vector<float> decoded(n_per_row);
    decode_source_row_to_f32(
        q8.data(),
        GGML_TYPE_Q8_0,
        n_per_row,
        decoded.data(),
        "layers.0.attn_q.weight");

    double max_error = 0.0;
    for (int64_t i = 0; i < n_per_row; ++i) {
        max_error = std::max(
            max_error,
            std::abs(static_cast<double>(source[static_cast<size_t>(i)] - decoded[static_cast<size_t>(i)])));
    }
    assert(max_error < 0.07);
}

void test_recipe_target_verification_counts_matching_changed_targets() {
    ggml_tensor token_tensor = {};
    token_tensor.type = GGML_TYPE_Q5_K;
    ggml_tensor output_tensor = {};
    output_tensor.type = GGML_TYPE_Q8_0;
    ggml_tensor norm_tensor = {};
    norm_tensor.type = GGML_TYPE_F32;

    std::vector<std::pair<std::string, ggml_tensor *>> tensor_map = {
        {"token_embd.weight", &token_tensor},
        {"output.weight", &output_tensor},
        {"output_norm.weight", &norm_tensor},
    };
    std::unordered_map<std::string, ggml_type> target_types = {
        {"token_embd.weight", GGML_TYPE_Q5_K},
        {"output.weight", GGML_TYPE_Q8_0},
        {"output_norm.weight", GGML_TYPE_F32},
    };
    std::unordered_map<std::string, ggml_type> source_types = {
        {"token_embd.weight", GGML_TYPE_BF16},
        {"output.weight", GGML_TYPE_BF16},
        {"output_norm.weight", GGML_TYPE_F32},
    };

    RecipeTargetVerification verification = verify_recipe_tensor_targets_in_map(
        tensor_map,
        target_types,
        source_types);

    assert(verification.requested_count == 2);
    assert(verification.verified_count == 2);
    assert(verification.mismatch_count == 0);
    assert(verification.first_mismatch.empty());
}

void test_recipe_target_verification_reports_mismatch() {
    ggml_tensor token_tensor = {};
    token_tensor.type = GGML_TYPE_Q8_0;

    std::vector<std::pair<std::string, ggml_tensor *>> tensor_map = {
        {"token_embd.weight", &token_tensor},
    };
    std::unordered_map<std::string, ggml_type> target_types = {
        {"token_embd.weight", GGML_TYPE_Q5_K},
    };
    std::unordered_map<std::string, ggml_type> source_types = {
        {"token_embd.weight", GGML_TYPE_BF16},
    };

    RecipeTargetVerification verification = verify_recipe_tensor_targets_in_map(
        tensor_map,
        target_types,
        source_types);

    assert(verification.requested_count == 1);
    assert(verification.verified_count == 0);
    assert(verification.mismatch_count == 1);
    assert(verification.first_mismatch.find("token_embd.weight expected Q5_K, loaded Q8_0") != std::string::npos);
}

void test_llama_ppl_chunks_score_second_half_of_complete_512_token_chunks() {
    const std::vector<LlamaPplChunk> chunks = build_llama_ppl_chunks(1200);

    assert(chunks.size() == 2);

    assert(chunks[0].begin == 0);
    assert(chunks[0].first_scored == 256);
    assert(chunks[0].end == 512);

    assert(chunks[1].begin == 512);
    assert(chunks[1].first_scored == 768);
    assert(chunks[1].end == 1024);

    uint64_t scored_tokens = 0;
    for (const LlamaPplChunk & chunk : chunks) {
        scored_tokens += static_cast<uint64_t>(chunk.end - chunk.first_scored - 1);
    }

    assert(scored_tokens == 510);
}

void test_llama_ppl_chunks_require_at_least_two_complete_contexts() {
    assert(build_llama_ppl_chunks(1023).empty());
    assert(build_llama_ppl_chunks(1024).size() == 2);
}

void test_llama_ppl_uncertainty_matches_upstream_formula() {
    const double estimate = llama_ppl_uncertainty(6.0, 14.0, 3);
    const double expected = std::exp(2.0) * std::sqrt((14.0 / 3.0 - 4.0) / 2.0);
    assert(std::abs(estimate - expected) < 1e-12);
}

void test_llama_mcq_common_prefix_stops_at_first_different_token() {
    const std::vector<std::vector<llama_token>> sequences = {
        {1, 7, 9, 10},
        {1, 7, 11, 12},
        {1, 7, 9, 13},
    };

    assert(find_common_token_prefix(sequences) == 2);
}

void test_llama_mcq_score_uses_token_average_logprob() {
    const double score = llama_mcq_choice_score(12.0, 3);

    assert(std::abs(score - -4.0) < 1e-12);
}

void test_recipe_test_cancellation_flag_can_reset_and_cancel() {
    ms_runtime_reset_recipe_test_cancel();
    assert(!recipe_test_cancel_requested());

    ms_runtime_cancel_recipe_test();
    assert(recipe_test_cancel_requested());

    ms_runtime_reset_recipe_test_cancel();
    assert(!recipe_test_cancel_requested());
}

void test_generation_context_allows_official_eval_prompts() {
    assert(session_context_tokens_for_generation(1024) >= 4096);
}

void test_context_generation_room_uses_context_window_not_request_max_tokens() {
    assert(context_generation_room(20'000, 2'000) == 17'999);
    assert(context_generation_room(20'000, 19'999) == 0);
}

void test_chat_template_formats_openai_messages() {
    const std::vector<std::pair<std::string, std::string>> messages = {
        {"system", "Answer carefully."},
        {"user", "What is GPQA?"},
    };
    auto templates = common_chat_templates_init(
        nullptr,
        "{{ bos_token }}{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endfor %}<|im_start|>assistant\n",
        "<s>",
        "</s>");
    const common_chat_params chat_params = format_chat_prompt_with_template(
        templates.get(),
        messages,
        true);
    const std::string & prompt = chat_params.prompt;

    assert(prompt.find("<|im_start|>system") != std::string::npos);
    assert(prompt.find("<|im_start|>user") != std::string::npos);
    assert(prompt.rfind("<|im_start|>assistant") != std::string::npos);
    assert(prompt.find("system: Answer carefully.") == std::string::npos);
}

void test_request_and_template_stop_strings_are_merged() {
    const std::vector<std::string> request_stops = {"</s>", "<|im_end|>"};
    const std::vector<std::string> template_stops = {"<|eot_id|>"};

    const std::vector<std::string> stops = merge_stop_strings(request_stops, template_stops);

    assert(stops.size() == 3);
    assert(stops[0] == "</s>");
    assert(stops[1] == "<|im_end|>");
    assert(stops[2] == "<|eot_id|>");
}

void test_generated_stop_suffix_is_consumed_in_loop() {
    std::string generated = "The answer is A<|im_end|>";

    const bool stopped = consume_generated_stop_suffix(generated, {"<|im_end|>"});

    assert(stopped);
    assert(generated == "The answer is A");
}

void test_chat_template_kwargs_json_matches_llama_server_shape() {
    const std::map<std::string, std::string> kwargs =
        chat_template_kwargs_from_json("{\"enable_thinking\":false,\"mode\":\"compact\"}");

    assert(kwargs.at("enable_thinking") == "false");
    assert(kwargs.at("mode") == "\"compact\"");
}

void test_missing_reasoning_format_defaults_to_llama_server_deepseek() {
    assert(reasoning_format_from_request(nullptr) == COMMON_REASONING_FORMAT_DEEPSEEK);
    assert(reasoning_format_from_request("") == COMMON_REASONING_FORMAT_DEEPSEEK);
}

void test_chat_sampling_params_normalize_context_windows_like_llama_server() {
    ms_chat_generation_params params = default_chat_generation_params(128);
    params.repeat_last_n = -1;
    params.dry_penalty_last_n = -1;

    const common_params_sampling sampling = common_sampling_from_chat_params(params, 4096);

    assert(sampling.penalty_last_n == 4096);
    assert(sampling.dry_penalty_last_n == 4096);
}

void test_chat_sampling_params_use_llama_server_sampler_chain() {
    const ms_chat_generation_params params = default_chat_generation_params(128);

    const common_params_sampling sampling = common_sampling_from_chat_params(params, 4096);

    const std::vector<common_sampler_type> expected = {
        COMMON_SAMPLER_TYPE_PENALTIES,
        COMMON_SAMPLER_TYPE_DRY,
        COMMON_SAMPLER_TYPE_TOP_N_SIGMA,
        COMMON_SAMPLER_TYPE_TOP_K,
        COMMON_SAMPLER_TYPE_TYPICAL_P,
        COMMON_SAMPLER_TYPE_TOP_P,
        COMMON_SAMPLER_TYPE_MIN_P,
        COMMON_SAMPLER_TYPE_XTC,
        COMMON_SAMPLER_TYPE_TEMPERATURE,
    };
    assert(sampling.samplers == expected);
}

void test_gemma4_generated_output_is_split_into_reasoning_and_visible_content() {
    common_chat_params chat_params = {};
    chat_params.format = COMMON_CHAT_FORMAT_PEG_GEMMA4;
    auto parser = build_chat_peg_parser([](common_chat_peg_builder & p) {
        auto thought = p.literal("<|channel>thought") + p.space() +
            p.reasoning(p.until("<channel|>")) + p.literal("<channel|>");
        auto content = p.content(p.rest());
        return thought + content + p.end();
    });
    chat_params.parser = parser.save();

    const ParsedChatOutput parsed = parse_generated_chat_output(
        "<|channel>thought\nprivate derivation<channel|>Final answer\nANSWER: C",
        chat_params,
        COMMON_REASONING_FORMAT_DEEPSEEK,
        false);

    assert(parsed.visible_text == "Final answer\nANSWER: C");
    assert(parsed.reasoning_text == "private derivation");
}

void test_runtime_version_advertises_chat_abi() {
    const std::string version = ms_runtime_version();
    assert(version.find("chat-abi=2") != std::string::npos);
}

void test_persistent_chat_session_loads_once_and_resets_context_per_completion() {
    const std::filesystem::path fixture =
        std::filesystem::path("models") / "test-subjects" / "Qwen_Qwen3-1.7B-bf16.gguf";
    if (!std::filesystem::exists(fixture)) {
        std::cout << "skipping persistent chat session counter test; fixture not found: "
                  << fixture.string()
                  << "\n";
        return;
    }

    gguf_init_params params = {};
    params.no_alloc = true;
    params.ctx = nullptr;
    std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(
        gguf_init_from_file(fixture.string().c_str(), params),
        gguf_free);
    assert(metadata != nullptr);

    std::vector<std::string> names;
    std::vector<std::string> quants;
    const int64_t tensor_count = gguf_get_n_tensors(metadata.get());
    names.reserve(static_cast<size_t>(tensor_count));
    quants.reserve(static_cast<size_t>(tensor_count));
    for (int64_t i = 0; i < tensor_count; ++i) {
        names.emplace_back(gguf_get_tensor_name(metadata.get(), i));
        quants.emplace_back(display_quant_type(gguf_get_tensor_type(metadata.get(), i)));
    }

    std::vector<ms_recipe_tensor_target> targets;
    targets.reserve(names.size());
    for (size_t i = 0; i < names.size(); ++i) {
        targets.push_back({names[i].c_str(), quants[i].c_str()});
    }

    std::vector<std::string> logs;
    ms_runtime_chat_session * session = nullptr;
    assert(ms_runtime_open_recipe_chat_session_with_progress(
        fixture.string().c_str(),
        targets.data(),
        targets.size(),
        1024,
        collect_runtime_log,
        &logs,
        &session) == 0);
    assert(session != nullptr);
    assert(std::find(
        logs.begin(),
        logs.end(),
        "Native runtime: loading model weights into memory") != logs.end());
    assert(std::find(
        logs.begin(),
        logs.end(),
        "Native runtime: model weights loaded") != logs.end());
    assert(std::find(
        logs.begin(),
        logs.end(),
        "Native runtime: chat context ready") != logs.end());

    const ms_chat_message messages[] = {
        {"user", "Reply with exactly one short word."},
    };
    char output[4096] = {};
    ms_chat_generation_result result = {};
    ms_chat_generation_params generation = default_chat_generation_params(4);
    generation.temperature = 0.0;
    const char * stop_strings[] = {"<|im_end|>"};
    char reasoning_output[4096] = {};
    assert(ms_runtime_generate_recipe_chat_session(
        session,
        messages,
        1,
        &generation,
        stop_strings,
        1,
        nullptr,
        nullptr,
        output,
        sizeof(output),
        reasoning_output,
        sizeof(reasoning_output),
        &result) == 0);
    assert(ms_runtime_generate_recipe_chat_session(
        session,
        messages,
        1,
        &generation,
        stop_strings,
        1,
        nullptr,
        nullptr,
        output,
        sizeof(output),
        reasoning_output,
        sizeof(reasoning_output),
        &result) == 0);

    ms_runtime_chat_session_counters counters = {};
    assert(ms_runtime_get_recipe_chat_session_counters(session, &counters) == 0);
    assert(counters.model_load_count == 1);
    assert(counters.context_reset_count == 2);
    assert(counters.completion_count == 2);

    ms_runtime_close_recipe_chat_session(session);
}

}

int main() {
    test_k_quant_recipe_targets_are_supported();
    test_legacy_quant_recipe_targets_are_supported();
    test_recipe_conversion_rejects_cross_family_quant_targets();
    test_quantized_source_rows_can_decode_to_f32();
    test_recipe_target_verification_counts_matching_changed_targets();
    test_recipe_target_verification_reports_mismatch();
    test_llama_ppl_chunks_score_second_half_of_complete_512_token_chunks();
    test_llama_ppl_chunks_require_at_least_two_complete_contexts();
    test_llama_ppl_uncertainty_matches_upstream_formula();
    test_llama_mcq_common_prefix_stops_at_first_different_token();
    test_llama_mcq_score_uses_token_average_logprob();
    test_recipe_test_cancellation_flag_can_reset_and_cancel();
    test_generation_context_allows_official_eval_prompts();
    test_context_generation_room_uses_context_window_not_request_max_tokens();
    test_chat_template_formats_openai_messages();
    test_request_and_template_stop_strings_are_merged();
    test_generated_stop_suffix_is_consumed_in_loop();
    test_chat_template_kwargs_json_matches_llama_server_shape();
    test_missing_reasoning_format_defaults_to_llama_server_deepseek();
    test_chat_sampling_params_normalize_context_windows_like_llama_server();
    test_chat_sampling_params_use_llama_server_sampler_chain();
    test_gemma4_generated_output_is_split_into_reasoning_and_visible_content();
    test_runtime_version_advertises_chat_abi();
    test_persistent_chat_session_loads_once_and_resets_context_per_completion();
    std::cout << "runtime quant tests passed\n";
    return EXIT_SUCCESS;
}
