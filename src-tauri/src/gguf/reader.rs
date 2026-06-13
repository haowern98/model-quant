use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use super::types::{GgufMetadata, ModelInfo, TensorInfo, TensorQuantPreflight};
use crate::quant::preflight::analyze_tensor_quant_preflight;

const GGUF_MAGIC: u32 = 0x46554747;

#[derive(Debug)]
pub enum GgufError {
    Io(std::io::Error),
    InvalidMagic,
    UnsupportedVersion(u32),
}

impl From<std::io::Error> for GgufError {
    fn from(e: std::io::Error) -> Self {
        GgufError::Io(e)
    }
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

fn read_string<R: Read>(reader: &mut R) -> Result<String, GgufError> {
    let len = read_u64(reader)? as usize;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).to_string())
}

fn read_u32<R: Read>(reader: &mut R) -> Result<u32, GgufError> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64<R: Read>(reader: &mut R) -> Result<u64, GgufError> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn classify_tensor(name: &str, layer_index: i32) -> &'static str {
    if name.contains("embed") || name.contains("tok_embd") || name.contains("token_embd") {
        "embedding"
    } else if name.contains("output.weight") && layer_index == -1 {
        "output"
    } else if name.contains("output_norm") || name.contains("final_norm") {
        "output_norm"
    } else if name.contains("norm") && layer_index >= 0 {
        "norm"
    } else if layer_index < 0 {
        "other"
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

// GGUF value types per spec
// 0=u8 1=i8 2=u16 3=i16 4=u32 5=i32 6=f32 7=bool
// 8=string 9=array 10=u64 11=i64 12=f64
fn skip_value<R: Read + Seek>(reader: &mut R, value_type: u32) -> Result<(), GgufError> {
    match value_type {
        0 | 1 | 7 => {
            reader.seek(SeekFrom::Current(1))?;
        }
        2 | 3 => {
            reader.seek(SeekFrom::Current(2))?;
        }
        4 | 5 | 6 => {
            reader.seek(SeekFrom::Current(4))?;
        }
        10 | 11 | 12 => {
            reader.seek(SeekFrom::Current(8))?;
        }
        8 => {
            let _ = read_string(reader)?;
        }
        9 => {
            let elem_type = read_u32(reader)?;
            let count = read_u64(reader)? as usize;
            for _ in 0..count {
                skip_value(reader, elem_type)?;
            }
        }
        _ => {} // unknown type, skip nothing (best effort)
    }
    Ok(())
}

fn ggml_type_name(t: u32) -> &'static str {
    match t {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        24 => "I8",
        25 => "I16",
        26 => "I32",
        27 => "I64",
        28 => "F64",
        29 => "IQ1_M",
        30 => "BF16",
        31 => "TQ1_0",
        32 => "TQ2_0",
        _ => "unknown",
    }
}

fn ggml_type_bits(t: u32) -> f32 {
    match t {
        0 => 32.0,
        1 | 30 => 16.0,
        2 | 3 => 4.0,
        6 | 7 => 5.0,
        8 | 9 | 15 | 24 => 8.0,
        10 => 2.5625,
        11 => 3.4375,
        12 => 4.5,
        13 => 5.5,
        14 => 6.5625,
        16 => 2.0625,
        17 => 2.3125,
        18 => 3.0625,
        19 => 1.5625,
        20 | 23 => 4.5,
        21 => 3.4375,
        22 => 2.5625,
        25 => 16.0,
        26 => 32.0,
        27 | 28 => 64.0,
        29 => 1.75,
        31 => 1.6875,
        32 => 2.625,
        _ => 16.0,
    }
}

pub fn parse_gguf(path: &Path) -> Result<ModelInfo, GgufError> {
    let mut file = File::open(path)?;

    let magic = read_u32(&mut file)?;
    if magic != GGUF_MAGIC {
        return Err(GgufError::InvalidMagic);
    }

    let version = read_u32(&mut file)?;
    if version != 3 && version != 2 {
        return Err(GgufError::UnsupportedVersion(version));
    }

    let tensor_count = read_u64(&mut file)? as usize;
    let metadata_kv_count = read_u64(&mut file)? as usize;

    let mut model_name = String::new();
    let mut architecture = String::new();

    for _ in 0..metadata_kv_count {
        let key = read_string(&mut file)?;
        let value_type = read_u32(&mut file)?;

        match key.as_str() {
            "general.name" | "general.architecture" => {
                let value = read_string(&mut file)?;
                if key == "general.name" {
                    model_name = value;
                } else if key == "general.architecture" {
                    architecture = value;
                }
            }
            _ => {
                skip_value(&mut file, value_type)?;
            }
        }
    }

    let mut tensors = Vec::with_capacity(tensor_count);
    let mut total_params: u64 = 0;

    for _ in 0..tensor_count {
        let name = read_string(&mut file)?;
        let n_dims = read_u32(&mut file)? as usize;
        let mut shape = Vec::with_capacity(n_dims);
        let mut elements: u64 = 1;

        for _ in 0..n_dims {
            let dim = read_u64(&mut file)?;
            shape.push(dim);
            elements = elements.saturating_mul(dim);
        }

        let ggml_type = read_u32(&mut file)?;
        let _tensor_data_offset = read_u64(&mut file)?;

        let quant_name = ggml_type_name(ggml_type);
        let bits = ggml_type_bits(ggml_type);
        let size_bytes = (elements as f64 * bits as f64 / 8.0).ceil() as u64;

        total_params = total_params.saturating_add(elements);

        let layer_index = extract_layer_index(&name);
        let layer_group = classify_tensor(&name, layer_index).to_string();

        let mut tensor = TensorInfo {
            name,
            shape,
            current_quant: quant_name.to_string(),
            size_bytes,
            layer_index,
            layer_group,
            quant_preflight: TensorQuantPreflight::pending(),
        };
        tensor.quant_preflight = analyze_tensor_quant_preflight(&tensor);
        tensors.push(tensor);
    }

    tensors.sort_by_key(|t| t.layer_index);

    let current_uniform_quant = most_common_quant(&tensors);

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

fn most_common_quant(tensors: &[TensorInfo]) -> String {
    let mut counts = std::collections::HashMap::<&str, usize>::new();
    for tensor in tensors {
        if matches!(tensor.current_quant.as_str(), "F32" | "F16" | "BF16") {
            continue;
        }
        *counts.entry(&tensor.current_quant).or_default() += 1;
    }

    counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(quant, _)| quant.to_string())
        .or_else(|| tensors.first().map(|t| t.current_quant.clone()))
        .unwrap_or_else(|| "unknown".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_tensor() {
        assert_eq!(classify_tensor("tok_embeddings.weight", -1), "embedding");
        assert_eq!(
            classify_tensor("layers.0.attention.wq.weight", 0),
            "attention"
        );
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
