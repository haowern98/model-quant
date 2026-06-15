use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(non_camel_case_types)]
pub enum QuantType {
    F32,
    BF16,
    F16,
    Q8_0,
    Q6_K,
    Q5_K,
    Q5_K_M,
    Q5_1,
    Q5_0,
    Q4_K,
    Q4_K_M,
    Q4_1,
    Q4_0,
    Q3_K,
    Q3_K_M,
    Q2_K,
}

impl QuantType {
    pub fn bits_per_weight(&self) -> f32 {
        match self {
            QuantType::F32 => 32.0,
            QuantType::BF16 => 16.0,
            QuantType::F16 => 16.0,
            QuantType::Q8_0 => 8.0,
            QuantType::Q6_K => 6.6,
            QuantType::Q5_K => 5.5,
            QuantType::Q5_K_M => 5.3,
            QuantType::Q5_1 => 6.0,
            QuantType::Q5_0 => 5.5,
            QuantType::Q4_K => 4.5,
            QuantType::Q4_K_M => 4.8,
            QuantType::Q4_1 => 5.0,
            QuantType::Q4_0 => 4.5,
            QuantType::Q3_K => 3.4,
            QuantType::Q3_K_M => 3.9,
            QuantType::Q2_K => 2.6,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QuantAssignment {
    pub tensor_name: String,
    pub quant_type: QuantType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecipeProfile {
    pub vram_estimate: f64,
    pub size_saved_vs_q8: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RecipeStatus {
    #[serde(rename = "draft")]
    Draft,
    #[serde(rename = "profiled")]
    Profiled,
    #[serde(rename = "applied")]
    Applied,
    #[serde(rename = "saved")]
    Saved,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecipeState {
    pub id: String,
    pub base_model: String,
    pub assignments: Vec<QuantAssignment>,
    pub profile: Option<RecipeProfile>,
    pub status: RecipeStatus,
}

impl RecipeState {
    pub fn new(base_model: String, tensor_names: Vec<String>, default_quant: QuantType) -> Self {
        let assignments = tensor_names
            .into_iter()
            .map(|tensor_name| QuantAssignment {
                tensor_name,
                quant_type: default_quant.clone(),
            })
            .collect();

        RecipeState {
            id: chrono::Utc::now().timestamp_millis().to_string(),
            base_model,
            assignments,
            profile: None,
            status: RecipeStatus::Draft,
        }
    }

    pub fn from_current_quants(base_model: String, tensors: Vec<(String, QuantType)>) -> Self {
        let assignments = tensors
            .into_iter()
            .map(|(tensor_name, quant_type)| QuantAssignment {
                tensor_name,
                quant_type,
            })
            .collect();

        RecipeState {
            id: chrono::Utc::now().timestamp_millis().to_string(),
            base_model,
            assignments,
            profile: None,
            status: RecipeStatus::Draft,
        }
    }

    pub fn assign_tensors(&mut self, names: &[String], quant_type: QuantType) {
        for name in names {
            if let Some(assign) = self.assignments.iter_mut().find(|a| &a.tensor_name == name) {
                assign.quant_type = quant_type.clone();
            }
        }
        self.status = RecipeStatus::Draft;
    }

    pub fn assign_all(&mut self, quant_type: QuantType) {
        for assign in &mut self.assignments {
            if should_bulk_assign(&assign.tensor_name, &quant_type) {
                assign.quant_type = quant_type.clone();
            }
        }
        self.status = RecipeStatus::Draft;
    }

    pub fn assign_by_pattern(&mut self, pattern: &str, quant_type: QuantType) {
        let is_match: Box<dyn Fn(&str) -> bool> = match pattern {
            "all_attn" => Box::new(is_attention_tensor),
            "all_ffn" => Box::new(is_ffn_tensor),
            "all_embeddings" => Box::new(is_embedding_tensor),
            _ => Box::new(|_: &str| true),
        };

        for assign in &mut self.assignments {
            if is_match(&assign.tensor_name) && should_bulk_assign(&assign.tensor_name, &quant_type)
            {
                assign.quant_type = quant_type.clone();
            }
        }
        self.status = RecipeStatus::Draft;
    }
}

fn should_bulk_assign(name: &str, quant_type: &QuantType) -> bool {
    !is_quantized_target(quant_type) || is_runtime_quantizable_tensor(name)
}

fn is_quantized_target(quant_type: &QuantType) -> bool {
    matches!(
        quant_type,
        QuantType::Q8_0
            | QuantType::Q6_K
            | QuantType::Q5_K
            | QuantType::Q5_K_M
            | QuantType::Q5_1
            | QuantType::Q5_0
            | QuantType::Q4_K
            | QuantType::Q4_K_M
            | QuantType::Q4_1
            | QuantType::Q4_0
            | QuantType::Q3_K
            | QuantType::Q3_K_M
            | QuantType::Q2_K
    )
}

fn is_runtime_quantizable_tensor(name: &str) -> bool {
    !contains_any(name, &["bias", "norm", "rope", "scale"])
}

fn is_attention_tensor(name: &str) -> bool {
    contains_any(
        name,
        &[
            "attention",
            "self_attn",
            "attn_",
            ".attn_",
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_output",
            "attn_norm",
        ],
    )
}

fn is_ffn_tensor(name: &str) -> bool {
    contains_any(
        name,
        &[
            "feed_forward",
            "ffn_",
            ".ffn_",
            "mlp",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
}

fn is_embedding_tensor(name: &str) -> bool {
    contains_any(
        name,
        &[
            "token_embd",
            "tok_embeddings",
            "embed_tokens",
            "embedding",
            "output.weight",
            "per_layer_token_embd",
        ],
    )
}

fn contains_any(name: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| name.contains(needle))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tensors() -> Vec<String> {
        vec![
            "tok_embeddings.weight".into(),
            "layers.0.attention.wq.weight".into(),
            "blk.0.attn_q.weight".into(),
            "blk.0.attn_norm.weight".into(),
            "layers.0.feed_forward.w1.weight".into(),
            "blk.0.ffn_gate.weight".into(),
            "blk.0.ffn_norm.weight".into(),
            "output.weight".into(),
            "output_norm.weight".into(),
        ]
    }

    #[test]
    fn test_new_recipe_default_quant() {
        let recipe = RecipeState::new("test.gguf".into(), make_tensors(), QuantType::Q4_K_M);
        assert_eq!(recipe.assignments.len(), 9);
        assert!(recipe
            .assignments
            .iter()
            .all(|a| a.quant_type == QuantType::Q4_K_M));
    }

    #[test]
    fn test_assign_single_tensor() {
        let mut recipe = RecipeState::new("test.gguf".into(), make_tensors(), QuantType::Q4_K_M);
        recipe.assign_tensors(&["layers.0.attention.wq.weight".into()], QuantType::Q6_K);
        let attn = recipe
            .assignments
            .iter()
            .find(|a| a.tensor_name == "layers.0.attention.wq.weight")
            .unwrap();
        assert_eq!(attn.quant_type, QuantType::Q6_K);
    }

    #[test]
    fn test_assign_by_pattern_attn() {
        let mut recipe = RecipeState::new("test.gguf".into(), make_tensors(), QuantType::Q4_K_M);
        recipe.assign_by_pattern("all_attn", QuantType::Q8_0);
        let attn = recipe
            .assignments
            .iter()
            .find(|a| a.tensor_name.contains("attention"))
            .unwrap();
        let gguf_attn = recipe
            .assignments
            .iter()
            .find(|a| a.tensor_name == "blk.0.attn_q.weight")
            .unwrap();
        let ffn = recipe
            .assignments
            .iter()
            .find(|a| a.tensor_name.contains("feed_forward"))
            .unwrap();
        let attn_norm = recipe
            .assignments
            .iter()
            .find(|a| a.tensor_name == "blk.0.attn_norm.weight")
            .unwrap();
        assert_eq!(attn.quant_type, QuantType::Q8_0);
        assert_eq!(gguf_attn.quant_type, QuantType::Q8_0);
        assert_eq!(attn_norm.quant_type, QuantType::Q4_K_M);
        assert_eq!(ffn.quant_type, QuantType::Q4_K_M);
    }

    #[test]
    fn test_assign_by_pattern_ffn() {
        let mut recipe = RecipeState::new("test.gguf".into(), make_tensors(), QuantType::Q4_K_M);
        recipe.assign_by_pattern("all_ffn", QuantType::Q8_0);
        let feed_forward = recipe
            .assignments
            .iter()
            .find(|a| a.tensor_name.contains("feed_forward"))
            .unwrap();
        let gguf_ffn = recipe
            .assignments
            .iter()
            .find(|a| a.tensor_name == "blk.0.ffn_gate.weight")
            .unwrap();
        let attn = recipe
            .assignments
            .iter()
            .find(|a| a.tensor_name == "blk.0.attn_q.weight")
            .unwrap();
        let ffn_norm = recipe
            .assignments
            .iter()
            .find(|a| a.tensor_name == "blk.0.ffn_norm.weight")
            .unwrap();
        assert_eq!(feed_forward.quant_type, QuantType::Q8_0);
        assert_eq!(gguf_ffn.quant_type, QuantType::Q8_0);
        assert_eq!(ffn_norm.quant_type, QuantType::Q4_K_M);
        assert_eq!(attn.quant_type, QuantType::Q4_K_M);
    }

    #[test]
    fn test_assign_by_pattern_embeddings() {
        let mut recipe = RecipeState::new(
            "test.gguf".into(),
            vec![
                "token_embd.weight".into(),
                "model.embed_tokens.weight".into(),
                "per_layer_token_embd.weight".into(),
                "output.weight".into(),
                "blk.0.attn_q.weight".into(),
            ],
            QuantType::Q4_K_M,
        );
        recipe.assign_by_pattern("all_embeddings", QuantType::Q8_0);

        for name in [
            "token_embd.weight",
            "model.embed_tokens.weight",
            "per_layer_token_embd.weight",
            "output.weight",
        ] {
            let assign = recipe
                .assignments
                .iter()
                .find(|a| a.tensor_name == name)
                .unwrap();
            assert_eq!(assign.quant_type, QuantType::Q8_0);
        }

        let attn = recipe
            .assignments
            .iter()
            .find(|a| a.tensor_name == "blk.0.attn_q.weight")
            .unwrap();
        assert_eq!(attn.quant_type, QuantType::Q4_K_M);
    }

    #[test]
    fn test_assign_all_quantized_preserves_runtime_only_tensors() {
        let mut recipe = RecipeState::new("test.gguf".into(), make_tensors(), QuantType::BF16);
        recipe.assign_all(QuantType::Q8_0);

        for name in [
            "tok_embeddings.weight",
            "layers.0.attention.wq.weight",
            "blk.0.attn_q.weight",
            "layers.0.feed_forward.w1.weight",
            "blk.0.ffn_gate.weight",
            "output.weight",
        ] {
            let assign = recipe
                .assignments
                .iter()
                .find(|a| a.tensor_name == name)
                .unwrap();
            assert_eq!(assign.quant_type, QuantType::Q8_0);
        }

        for name in [
            "blk.0.attn_norm.weight",
            "blk.0.ffn_norm.weight",
            "output_norm.weight",
        ] {
            let assign = recipe
                .assignments
                .iter()
                .find(|a| a.tensor_name == name)
                .unwrap();
            assert_eq!(assign.quant_type, QuantType::BF16);
        }
    }

    #[test]
    fn test_recipe_json_roundtrip() {
        let recipe = RecipeState::new("test.gguf".into(), make_tensors(), QuantType::Q4_K_M);
        let json = serde_json::to_string(&recipe).unwrap();
        let parsed: RecipeState = serde_json::from_str(&json).unwrap();
        assert_eq!(recipe.id, parsed.id);
        assert_eq!(recipe.assignments.len(), parsed.assignments.len());
    }
}
