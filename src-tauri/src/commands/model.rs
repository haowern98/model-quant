use std::path::PathBuf;
use tauri::State;
use std::sync::Mutex;

use crate::gguf::reader::parse_gguf;
use crate::gguf::types::ModelInfo;
use crate::commands::quant::RecipeStore;
use crate::quant::recipe::{QuantType, RecipeState};

pub struct ModelState(pub Mutex<Option<ModelInfo>>);

#[tauri::command]
pub async fn open_model(
    path: String,
    model_state: State<'_, ModelState>,
    recipe_state: State<'_, RecipeStore>,
) -> Result<ModelInfo, String> {
    let model_path = PathBuf::from(&path);
    let model = parse_gguf(&model_path).map_err(|e| e.to_string())?;

    {
        let mut guard = model_state.0.lock().map_err(|e| e.to_string())?;
        *guard = Some(model.clone());
    }

    {
        let tensor_names = model.tensors.iter().map(|t| t.name.clone()).collect();
        let default_quant = parse_default_quant(&model.current_uniform_quant);
        let mut guard = recipe_state.0.lock().map_err(|e| e.to_string())?;
        *guard = Some(RecipeState::new(path, tensor_names, default_quant));
    }

    Ok(model)
}

#[tauri::command]
pub async fn get_tensors(state: State<'_, ModelState>) -> Result<Vec<crate::gguf::types::TensorInfo>, String> {
    let guard = state.0.lock().map_err(|e| e.to_string())?;
    match &*guard {
        Some(model) => Ok(model.tensors.clone()),
        None => Err("No model loaded".into()),
    }
}

fn parse_default_quant(value: &str) -> QuantType {
    match value {
        "F16" => QuantType::F16,
        "Q8_0" => QuantType::Q8_0,
        "Q6_K" => QuantType::Q6_K,
        "Q5_K_M" => QuantType::Q5_K_M,
        "Q4_K_M" | "Q4_K" => QuantType::Q4_K_M,
        "Q3_K_M" | "Q3_K" => QuantType::Q3_K_M,
        "Q2_K" => QuantType::Q2_K,
        _ => QuantType::Q4_K_M,
    }
}
