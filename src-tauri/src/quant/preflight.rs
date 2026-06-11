use crate::gguf::types::{TensorInfo, TensorQuantPreflight};

struct TargetQuant {
    name: &'static str,
    bits_per_weight: f32,
    block_size: Option<u64>,
}

const DIRECT_TARGET_QUANTS: &[TargetQuant] = &[
    TargetQuant {
        name: "F32",
        bits_per_weight: 32.0,
        block_size: None,
    },
    TargetQuant {
        name: "BF16",
        bits_per_weight: 16.0,
        block_size: None,
    },
    TargetQuant {
        name: "F16",
        bits_per_weight: 16.0,
        block_size: None,
    },
    TargetQuant {
        name: "Q8_0",
        bits_per_weight: 8.0,
        block_size: Some(32),
    },
    TargetQuant {
        name: "Q6_K",
        bits_per_weight: 6.6,
        block_size: Some(256),
    },
    TargetQuant {
        name: "Q5_K",
        bits_per_weight: 5.5,
        block_size: Some(256),
    },
    TargetQuant {
        name: "Q4_K",
        bits_per_weight: 4.5,
        block_size: Some(256),
    },
    TargetQuant {
        name: "Q3_K",
        bits_per_weight: 3.4,
        block_size: Some(256),
    },
    TargetQuant {
        name: "Q2_K",
        bits_per_weight: 2.6,
        block_size: Some(256),
    },
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

    let Some(current_bits) = bits_per_weight(&tensor.current_quant) else {
        return blocked("unsupported current quant type");
    };

    let allowed_target_quants = DIRECT_TARGET_QUANTS
        .iter()
        .filter(|target| target.bits_per_weight <= current_bits)
        .filter(|target| {
            target
                .block_size
                .map_or(true, |block| tensor.shape[0] % block == 0)
        })
        .map(|target| target.name.to_string())
        .collect::<Vec<_>>();

    if allowed_target_quants.is_empty() {
        return blocked("no equal-or-smaller target quant fits this tensor row");
    }

    TensorQuantPreflight {
        can_quantize: true,
        allowed_target_quants,
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
