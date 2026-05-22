use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(non_camel_case_types)]
pub enum QuantType {
    F16,
    Q8_0,
    Q6_K,
    Q5_K_M,
    Q4_K_M,
    Q3_K_M,
    Q2_K,
}

impl QuantType {
    pub fn bits_per_weight(&self) -> f32 {
        match self {
            QuantType::F16 => 16.0,
            QuantType::Q8_0 => 8.0,
            QuantType::Q6_K => 6.6,
            QuantType::Q5_K_M => 5.3,
            QuantType::Q4_K_M => 4.8,
            QuantType::Q3_K_M => 3.9,
            QuantType::Q2_K => 2.6,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantAssignment {
    pub tensor_name: String,
    pub quant_type: QuantType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
            assign.quant_type = quant_type.clone();
        }
        self.status = RecipeStatus::Draft;
    }

    pub fn assign_by_pattern(&mut self, pattern: &str, quant_type: QuantType) {
        let is_match: Box<dyn Fn(&str) -> bool> = match pattern {
            "all_attn" => Box::new(|n: &str| n.contains("attention")),
            "all_ffn" => Box::new(|n: &str| n.contains("feed_forward") || n.contains("ffn")),
            "all_embeddings" => Box::new(|n: &str| n.contains("embedding") || n.contains("output.weight")),
            _ => Box::new(|_: &str| true),
        };

        for assign in &mut self.assignments {
            if is_match(&assign.tensor_name) {
                assign.quant_type = quant_type.clone();
            }
        }
        self.status = RecipeStatus::Draft;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tensors() -> Vec<String> {
        vec![
            "tok_embeddings.weight".into(),
            "layers.0.attention.wq.weight".into(),
            "layers.0.feed_forward.w1.weight".into(),
            "output.weight".into(),
        ]
    }

    #[test]
    fn test_new_recipe_default_quant() {
        let recipe = RecipeState::new("test.gguf".into(), make_tensors(), QuantType::Q4_K_M);
        assert_eq!(recipe.assignments.len(), 4);
        assert!(recipe.assignments.iter().all(|a| a.quant_type == QuantType::Q4_K_M));
    }

    #[test]
    fn test_assign_single_tensor() {
        let mut recipe = RecipeState::new("test.gguf".into(), make_tensors(), QuantType::Q4_K_M);
        recipe.assign_tensors(&["layers.0.attention.wq.weight".into()], QuantType::Q6_K);
        let attn = recipe.assignments.iter().find(|a| a.tensor_name == "layers.0.attention.wq.weight").unwrap();
        assert_eq!(attn.quant_type, QuantType::Q6_K);
    }

    #[test]
    fn test_assign_by_pattern_attn() {
        let mut recipe = RecipeState::new("test.gguf".into(), make_tensors(), QuantType::Q4_K_M);
        recipe.assign_by_pattern("all_attn", QuantType::Q8_0);
        let attn = recipe.assignments.iter().find(|a| a.tensor_name.contains("attention")).unwrap();
        let ffn = recipe.assignments.iter().find(|a| a.tensor_name.contains("feed_forward")).unwrap();
        assert_eq!(attn.quant_type, QuantType::Q8_0);
        assert_eq!(ffn.quant_type, QuantType::Q4_K_M);
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
