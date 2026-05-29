use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GgufMetadata {
    pub name: String,
    pub architecture: String,
    pub total_params: u64,
    pub total_size_fp16: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub current_quant: String,
    pub size_bytes: u64,
    pub layer_index: i32,
    pub layer_group: String,
    pub quant_preflight: TensorQuantPreflight,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct TensorQuantPreflight {
    pub can_quantize: bool,
    pub allowed_target_quants: Vec<String>,
    pub blocked_reason: Option<String>,
}

impl TensorQuantPreflight {
    pub fn pending() -> Self {
        Self {
            can_quantize: false,
            allowed_target_quants: Vec::new(),
            blocked_reason: Some("quantization preflight has not run".to_string()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInfo {
    pub metadata: GgufMetadata,
    pub tensors: Vec<TensorInfo>,
    pub current_uniform_quant: String,
    pub total_size_bytes: u64,
}
