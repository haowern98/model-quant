#include "../src/model_surgery_runtime.cpp"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>

namespace {

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

void test_rolling_ppl_windows_score_each_target_token_once() {
    const std::vector<RollingPplWindow> windows = build_rolling_ppl_windows(269, 128);

    assert(windows.size() == 4);

    assert(windows[0].begin == 0);
    assert(windows[0].target_begin == 1);
    assert(windows[0].end == 128);

    assert(windows[1].begin == 64);
    assert(windows[1].target_begin == 128);
    assert(windows[1].end == 192);

    assert(windows[2].begin == 128);
    assert(windows[2].target_begin == 192);
    assert(windows[2].end == 256);

    assert(windows[3].begin == 141);
    assert(windows[3].target_begin == 256);
    assert(windows[3].end == 269);

    uint64_t scored_tokens = 0;
    for (const RollingPplWindow & window : windows) {
        scored_tokens += static_cast<uint64_t>(window.end - window.target_begin);
    }

    assert(scored_tokens == 268);
}

}

int main() {
    test_k_quant_recipe_targets_are_supported();
    test_quantized_source_rows_can_decode_to_f32();
    test_recipe_target_verification_counts_matching_changed_targets();
    test_recipe_target_verification_reports_mismatch();
    test_rolling_ppl_windows_score_each_target_token_once();
    std::cout << "runtime quant tests passed\n";
    return EXIT_SUCCESS;
}
