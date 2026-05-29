use crate::gguf::types::{TensorInfo, TensorQuantPreflight};

const DIRECT_TARGET_QUANTS: &[&str] = &[
    "F32", "BF16", "F16", "Q8_0", "Q6_K", "Q5_K", "Q4_K", "Q3_K", "Q2_K",
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

    TensorQuantPreflight {
        can_quantize: true,
        allowed_target_quants: DIRECT_TARGET_QUANTS
            .iter()
            .map(|quant| quant.to_string())
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
