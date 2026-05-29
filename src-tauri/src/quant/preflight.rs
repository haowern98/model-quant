use crate::gguf::types::{TensorInfo, TensorQuantPreflight};

const DIRECT_TARGET_QUANTS: &[(&str, f32)] = &[
    ("F32", 32.0),
    ("BF16", 16.0),
    ("F16", 16.0),
    ("Q8_0", 8.0),
    ("Q6_K", 6.6),
    ("Q5_K", 5.5),
    ("Q4_K", 4.5),
    ("Q3_K", 3.4),
    ("Q2_K", 2.6),
];

pub fn analyze_tensor_quant_preflight(tensor: &TensorInfo) -> TensorQuantPreflight {
    if tensor.shape.is_empty() || tensor.shape.iter().any(|dim| *dim == 0) {
        return blocked("tensor shape is empty or invalid");
    }

    if tensor.shape.len() < 2 {
        return blocked("1D tensors are not quantizable weight matrices");
    }

    let name = tensor.name.to_ascii_lowercase();
    if contains_any(&name, &["bias", "norm", "rope", "scale"]) {
        return blocked("runtime or normalization tensors are not quantizable weight matrices");
    }

    if tensor.shape[0] % 32 != 0 {
        return blocked("tensor row width is not divisible by Q8_0 block size 32");
    }

    let Some(current_bits) = bits_per_weight(&tensor.current_quant) else {
        return blocked("unsupported current quant type");
    };

    TensorQuantPreflight {
        can_quantize: true,
        allowed_target_quants: DIRECT_TARGET_QUANTS
            .iter()
            .filter(|(_, target_bits)| *target_bits <= current_bits)
            .map(|(quant, _)| quant.to_string())
            .collect(),
        blocked_reason: None,
    }
}

fn blocked(reason: &str) -> TensorQuantPreflight {
    TensorQuantPreflight {
        can_quantize: false,
        allowed_target_quants: Vec::new(),
        blocked_reason: Some(reason.to_string()),
    }
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| haystack.contains(needle))
}

fn bits_per_weight(quant: &str) -> Option<f32> {
    match quant {
        "F32" => Some(32.0),
        "BF16" | "F16" => Some(16.0),
        "Q8_0" => Some(8.0),
        "Q6_K" => Some(6.6),
        "Q5_K" => Some(5.5),
        "Q5_K_M" => Some(5.3),
        "Q4_K_M" => Some(4.8),
        "Q4_K" => Some(4.5),
        "Q3_K_M" => Some(3.9),
        "Q3_K" => Some(3.4),
        "Q2_K" => Some(2.6),
        _ => None,
    }
}
