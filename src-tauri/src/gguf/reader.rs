use std::fs::File;
use std::io::Read;
use std::path::Path;

use super::types::{GgufMetadata, ModelInfo, TensorInfo};

const GGUF_MAGIC: u32 = 0x46554747;

#[derive(Debug)]
pub enum GgufError {
    Io(std::io::Error),
    InvalidMagic,
    UnsupportedVersion(u32),
}

impl From<std::io::Error> for GgufError {
    fn from(e: std::io::Error) -> Self { GgufError::Io(e) }
}

impl std::fmt::Display for GgufError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GgufError::Io(e) => write!(f, "IO error: {}", e),
            GgufError::InvalidMagic => write!(f, "Not a valid GGUF file"),
            GgufError::UnsupportedVersion(v) => write!(f, "Unsupported GGUF version: {}", v),
        }
    }
}

fn read_string(buf: &[u8], offset: &mut usize) -> String {
    let len = u64::from_le_bytes(buf[*offset..*offset + 8].try_into().unwrap()) as usize;
    *offset += 8;
    let s = String::from_utf8_lossy(&buf[*offset..*offset + len]).to_string();
    *offset += len;
    s
}

fn read_u32(buf: &[u8], offset: &mut usize) -> u32 {
    let v = u32::from_le_bytes(buf[*offset..*offset + 4].try_into().unwrap());
    *offset += 4;
    v
}

fn read_u64(buf: &[u8], offset: &mut usize) -> u64 {
    let v = u64::from_le_bytes(buf[*offset..*offset + 8].try_into().unwrap());
    *offset += 8;
    v
}

fn classify_tensor(name: &str, layer_index: i32) -> &'static str {
    if name.contains("embedding") || name.contains("tok_embeddings") {
        "embedding"
    } else if name.contains("output_norm") || name.contains("norm") {
        "output_norm"
    } else if name.contains("output") && layer_index == -1 {
        "output"
    } else {
        "attention"
    }
}

fn extract_layer_index(name: &str) -> i32 {
    for prefix in &["layers.", "blk."] {
        if let Some(rest) = name.strip_prefix(prefix) {
            if let Some(dot) = rest.find('.') {
                if let Ok(idx) = rest[..dot].parse::<i32>() {
                    return idx;
                }
            }
        }
    }
    -1
}

fn ggml_type_name(t: u32) -> &'static str {
    match t {
        0 => "F32", 1 => "F16",
        7 => "Q4_0", 8 => "Q4_1", 10 => "Q8_0",
        12 => "Q2_K", 14 => "Q3_K_M", 16 => "Q5_K_M",
        17 => "Q6_K", 18 => "Q4_K_M",
        _ => "unknown",
    }
}

fn ggml_type_bits(t: u32) -> u32 {
    match t {
        0 => 32, 1 => 16,
        7 | 8 | 12 => 4, 10 => 8,
        14 => 3, 16 => 5, 17 => 6, 18 => 4,
        _ => 16,
    }
}

pub fn parse_gguf(path: &Path) -> Result<ModelInfo, GgufError> {
    let mut file = File::open(path)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;

    let mut offset = 0usize;

    let magic = read_u32(&buf, &mut offset);
    if magic != GGUF_MAGIC {
        return Err(GgufError::InvalidMagic);
    }

    let version = read_u32(&buf, &mut offset);
    if version != 3 && version != 2 {
        return Err(GgufError::UnsupportedVersion(version));
    }

    let tensor_count = read_u64(&buf, &mut offset) as usize;
    let metadata_kv_count = read_u64(&buf, &mut offset) as usize;

    let mut model_name = String::new();
    let mut architecture = String::new();

    for _ in 0..metadata_kv_count {
        let key = read_string(&buf, &mut offset);
        let value_type = read_u32(&buf, &mut offset);

        match key.as_str() {
            "general.name" | "general.architecture" => {
                if key == "general.name" { model_name = read_string(&buf, &mut offset); }
                else if key == "general.architecture" { architecture = read_string(&buf, &mut offset); }
                else { let _ = read_string(&buf, &mut offset); }
            }
            _ => {
                match value_type {
                    0..=7 => { offset += 1; }
                    8 | 9 => { offset += 2; }
                    10 | 11 => { offset += 4; }
                    12 | 13 | 14 => { offset += 8; }
                    15 | 16 => { let _ = read_string(&buf, &mut offset); }
                    _ => {}
                }
            }
        }
    }

    let mut tensors = Vec::with_capacity(tensor_count);
    let mut total_params: u64 = 0;

    for _ in 0..tensor_count {
        let name = read_string(&buf, &mut offset);
        let n_dims = read_u32(&buf, &mut offset) as usize;
        let mut shape = Vec::with_capacity(n_dims);
        let mut elements: u64 = 1;

        for _ in 0..n_dims {
            let dim = read_u64(&buf, &mut offset);
            shape.push(dim);
            elements = elements.saturating_mul(dim);
        }

        let ggml_type = read_u32(&buf, &mut offset);
        offset += 4; // tensor data offset

        let quant_name = ggml_type_name(ggml_type);
        let bits = ggml_type_bits(ggml_type);
        let size_bytes = elements.saturating_mul(bits as u64) / 8;

        total_params = total_params.saturating_add(elements);

        let layer_index = extract_layer_index(&name);
        let layer_group = classify_tensor(&name, layer_index).to_string();

        tensors.push(TensorInfo {
            name,
            shape,
            current_quant: quant_name.to_string(),
            size_bytes,
            layer_index,
            layer_group,
        });
    }

    tensors.sort_by_key(|t| t.layer_index);

    let current_uniform_quant = tensors.first()
        .map(|t| t.current_quant.clone())
        .unwrap_or_else(|| "unknown".to_string());

    let total_size_bytes = tensors.iter().map(|t| t.size_bytes).sum();

    Ok(ModelInfo {
        metadata: GgufMetadata {
            name: model_name,
            architecture,
            total_params,
            total_size_fp16: total_params.saturating_mul(2),
        },
        tensors,
        current_uniform_quant,
        total_size_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_tensor() {
        assert_eq!(classify_tensor("tok_embeddings.weight", -1), "embedding");
        assert_eq!(classify_tensor("layers.0.attention.wq.weight", 0), "attention");
        assert_eq!(classify_tensor("output_norm.weight", -1), "output_norm");
        assert_eq!(classify_tensor("output.weight", -1), "output");
    }

    #[test]
    fn test_extract_layer_index() {
        assert_eq!(extract_layer_index("layers.0.attention.wq.weight"), 0);
        assert_eq!(extract_layer_index("layers.31.feed_forward.w3.weight"), 31);
        assert_eq!(extract_layer_index("tok_embeddings.weight"), -1);
    }
}
