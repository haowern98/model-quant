use model_surgery::gguf::types::{TensorInfo, TensorQuantPreflight};
use model_surgery::quant::preflight::analyze_tensor_quant_preflight;

fn tensor(name: &str, shape: Vec<u64>, layer_group: &str) -> TensorInfo {
    tensor_with_quant(name, shape, layer_group, "BF16")
}

fn tensor_with_quant(
    name: &str,
    shape: Vec<u64>,
    layer_group: &str,
    current_quant: &str,
) -> TensorInfo {
    TensorInfo {
        name: name.to_string(),
        shape,
        current_quant: current_quant.to_string(),
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
fn preflight_filters_q8_tensor_to_q8_and_smaller_targets() {
    let result = analyze_tensor_quant_preflight(&tensor_with_quant(
        "layers.0.attn_q.weight",
        vec![2048, 2048],
        "attention",
        "Q8_0",
    ));

    assert!(result.can_quantize);
    assert_eq!(
        result.allowed_target_quants,
        vec![
            "Q8_0".to_string(),
            "Q6_K".to_string(),
            "Q5_K".to_string(),
            "Q4_K".to_string(),
            "Q3_K".to_string(),
            "Q2_K".to_string(),
        ]
    );
}

#[test]
fn preflight_filters_q4_tensor_to_q4_and_smaller_targets() {
    let result = analyze_tensor_quant_preflight(&tensor_with_quant(
        "layers.0.attn_q.weight",
        vec![2048, 2048],
        "attention",
        "Q4_K",
    ));

    assert!(result.can_quantize);
    assert_eq!(
        result.allowed_target_quants,
        vec!["Q4_K".to_string(), "Q3_K".to_string(), "Q2_K".to_string()]
    );
}

#[test]
fn preflight_filters_q2_tensor_to_q2_only() {
    let result = analyze_tensor_quant_preflight(&tensor_with_quant(
        "layers.0.attn_q.weight",
        vec![2048, 2048],
        "attention",
        "Q2_K",
    ));

    assert!(result.can_quantize);
    assert_eq!(result.allowed_target_quants, vec!["Q2_K".to_string()]);
}

#[test]
fn preflight_blocks_rows_that_do_not_fit_q8_blocks() {
    let result = analyze_tensor_quant_preflight(&tensor_with_quant(
        "layers.0.attn_q.weight",
        vec![33, 2048],
        "attention",
        "Q8_0",
    ));

    assert!(!result.can_quantize);
    assert!(result.allowed_target_quants.is_empty());
    assert_eq!(
        result.blocked_reason.as_deref(),
        Some("no equal-or-smaller target quant fits this tensor row")
    );
}

#[test]
fn preflight_checks_each_target_quant_block_size() {
    let result = analyze_tensor_quant_preflight(&tensor(
        "layers.0.attn_q.weight",
        vec![128, 2048],
        "attention",
    ));

    assert!(result.can_quantize);
    assert_eq!(
        result.allowed_target_quants,
        vec![
            "BF16".to_string(),
            "F16".to_string(),
            "Q8_0".to_string(),
        ]
    );
}

#[test]
fn preflight_keeps_only_current_quant_when_no_smaller_target_fits() {
    let result = analyze_tensor_quant_preflight(&tensor_with_quant(
        "layers.0.attn_q.weight",
        vec![128, 2048],
        "attention",
        "Q8_0",
    ));

    assert!(result.can_quantize);
    assert_eq!(result.allowed_target_quants, vec!["Q8_0".to_string()]);
}
