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

}

int main() {
    test_k_quant_recipe_targets_are_supported();
    test_quantized_source_rows_can_decode_to_f32();
    std::cout << "runtime quant tests passed\n";
    return EXIT_SUCCESS;
}
