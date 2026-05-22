use std::path::PathBuf;
use tauri::State;
use std::sync::Mutex;

use crate::gguf::reader::parse_gguf;
use crate::gguf::types::ModelInfo;

pub struct ModelState(pub Mutex<Option<ModelInfo>>);

#[tauri::command]
pub async fn open_model(path: String) -> Result<ModelInfo, String> {
    let path = PathBuf::from(&path);
    parse_gguf(&path).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn get_tensors(state: State<'_, ModelState>) -> Result<Vec<crate::gguf::types::TensorInfo>, String> {
    let guard = state.0.lock().map_err(|e| e.to_string())?;
    match &*guard {
        Some(model) => Ok(model.tensors.clone()),
        None => Err("No model loaded".into()),
    }
}
