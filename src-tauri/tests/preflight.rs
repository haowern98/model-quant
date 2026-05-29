use model_surgery::gguf::types::{TensorInfo, TensorQuantPreflight};
use model_surgery::quant::preflight::analyze_tensor_quant_preflight;

fn tensor(name: &str, shape: Vec<u64>, layer_group: &str) -> TensorInfo {
    TensorInfo {
        name: name.to_string(),
        shape,
        current_quant: "BF16".to_string(),
        size_bytes: 0,
        layer_index: -1,
        layer_group: layer_group.to_string(),
        quant_preflight: TensorQuantPreflight::pending(),
    }
}

#[test]
fn preflight_blocks_norm_vectors() {
    let result = analyze_tensor_quant_preflight(&tensor(
        "output_norm.weight",
        vec![2048],
        "output_norm",
    ));

    assert!(!result.can_quantize);
    assert!(result.allowed_target_quants.is_empty());
    assert_eq!(
        result.blocked_reason.as_deref(),
        Some("1D tensors are not quantizable weight matrices")
    );
}

#[test]
fn preflight_allows_matrix_weights() {
    let result = analyze_tensor_quant_preflight(&tensor(
        "output.weight",
        vec![2048, 151936],
        "output",
    ));

    assert!(result.can_quantize);
    assert_eq!(
        result.allowed_target_quants,
        vec![
            "F32".to_string(),
            "BF16".to_string(),
            "F16".to_string(),
            "Q8_0".to_string(),
            "Q6_K".to_string(),
            "Q5_K".to_string(),
            "Q4_K".to_string(),
            "Q3_K".to_string(),
            "Q2_K".to_string(),
        ]
    );
    assert_eq!(result.blocked_reason, None);
}

#[test]
fn preflight_blocks_rows_that_do_not_fit_q8_blocks() {
    let result = analyze_tensor_quant_preflight(&tensor(
        "layers.0.attn_q.weight",
        vec![33, 2048],
        "attention",
    ));

    assert!(!result.can_quantize);
    assert!(result.allowed_target_quants.is_empty());
    assert_eq!(
        result.blocked_reason.as_deref(),
        Some("tensor row width is not divisible by Q8_0 block size 32")
    );
}
