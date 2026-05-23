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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInfo {
    pub metadata: GgufMetadata,
    pub tensors: Vec<TensorInfo>,
    pub current_uniform_quant: String,
    pub total_size_bytes: u64,
}
