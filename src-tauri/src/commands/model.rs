use std::path::PathBuf;
use std::sync::Mutex;
use tauri::State;

use crate::commands::quant::RecipeStore;
use crate::gguf::reader::parse_gguf;
use crate::gguf::types::ModelInfo;
use crate::quant::recipe::RecipeState;

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
        let tensor_quants = model
            .tensors
            .iter()
            .map(|t| (t.name.clone(), t.current_quant.clone()))
            .collect();
        let mut guard = recipe_state.0.lock().map_err(|e| e.to_string())?;
        *guard = Some(RecipeState::from_current_quants(path, tensor_quants));
    }

    Ok(model)
}

#[tauri::command]
pub async fn get_tensors(
    state: State<'_, ModelState>,
) -> Result<Vec<crate::gguf::types::TensorInfo>, String> {
    let guard = state.0.lock().map_err(|e| e.to_string())?;
    match &*guard {
        Some(model) => Ok(model.tensors.clone()),
        None => Err("No model loaded".into()),
    }
}
