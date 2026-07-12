use std::path::PathBuf;
use std::sync::Mutex;
use tauri::State;

use crate::commands::quant::RecipeStore;
use crate::ffi::runtime_bindings;
use crate::gguf::reader::parse_gguf;
use crate::gguf::types::ModelInfo;
use crate::quant::recipe::{QuantType, RecipeState};

const TENSOR_VALUES_MAX_WINDOW: u64 = 4096;

pub struct LoadedModel {
    path: PathBuf,
    pub(crate) info: ModelInfo,
}

pub struct ModelState(pub Mutex<Option<LoadedModel>>);
pub struct ProjectorState(pub Mutex<Option<LoadedModel>>);

fn is_standalone_projector_gguf<'a>(
    architecture: &str,
    tensor_names: impl Iterator<Item = &'a str>,
) -> bool {
    let mut has_language_model = false;
    let mut has_projector = architecture.eq_ignore_ascii_case("clip");

    for name in tensor_names {
        has_language_model |= matches!(
            name,
            name if name.starts_with("token_embd.")
                || name.starts_with("tok_embeddings.")
                || name.starts_with("model.embed_tokens.")
                || name.starts_with("language_model.model.embed_tokens.")
        );
        has_projector |= name.starts_with("v.")
            || name.starts_with("mm.")
            || name.starts_with("clip.")
            || name.starts_with("vision_model.")
            || name.starts_with("vision_tower.")
            || name.starts_with("multi_modal_projector.");
    }

    has_projector && !has_language_model
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TensorValuesPreview {
    values: Vec<f32>,
    rows: u64,
    cols: u64,
    total_rows: u64,
    total_cols: u64,
}

#[tauri::command]
pub async fn open_model(
    path: String,
    model_state: State<'_, ModelState>,
    recipe_state: State<'_, RecipeStore>,
) -> Result<ModelInfo, String> {
    let model_path = PathBuf::from(&path);
    let model = parse_gguf(&model_path).map_err(|e| e.to_string())?;
    if is_standalone_projector_gguf(
        &model.metadata.architecture,
        model.tensors.iter().map(|tensor| tensor.name.as_str()),
    ) {
        return Err("This is an MMPROJ GGUF. Add it in the MMPROJ section.".to_string());
    }

    {
        let mut guard = model_state.0.lock().map_err(|e| e.to_string())?;
        *guard = Some(LoadedModel {
            path: model_path,
            info: model.clone(),
        });
    }

    {
        let tensor_quants = model
            .tensors
            .iter()
            .map(|t| (t.name.clone(), parse_default_quant(&t.current_quant)))
            .collect();
        let mut guard = recipe_state.0.lock().map_err(|e| e.to_string())?;
        *guard = Some(RecipeState::from_current_quants(path, tensor_quants));
    }

    Ok(model)
}

#[tauri::command]
pub async fn open_projector(
    path: String,
    projector_state: State<'_, ProjectorState>,
) -> Result<ModelInfo, String> {
    let projector_path = PathBuf::from(&path);
    let model = parse_gguf(&projector_path).map_err(|e| e.to_string())?;
    if !is_standalone_projector_gguf(
        &model.metadata.architecture,
        model.tensors.iter().map(|tensor| tensor.name.as_str()),
    ) {
        return Err("This is a model GGUF. Add it in the model GGUF section.".to_string());
    }

    let mut guard = projector_state.0.lock().map_err(|e| e.to_string())?;
    *guard = Some(LoadedModel {
        path: projector_path,
        info: model.clone(),
    });
    Ok(model)
}

#[tauri::command]
pub async fn close_projector(projector_state: State<'_, ProjectorState>) -> Result<(), String> {
    *projector_state.0.lock().map_err(|e| e.to_string())? = None;
    Ok(())
}

#[tauri::command]
pub async fn get_tensors(
    state: State<'_, ModelState>,
) -> Result<Vec<crate::gguf::types::TensorInfo>, String> {
    let guard = state.0.lock().map_err(|e| e.to_string())?;
    match &*guard {
        Some(model) => Ok(model.info.tensors.clone()),
        None => Err("No model loaded".into()),
    }
}

#[tauri::command]
pub async fn get_tensor_values(
    state: State<'_, ModelState>,
    projector_state: State<'_, ProjectorState>,
    source: Option<String>,
    tensor_name: String,
    row_offset: u64,
    col_offset: u64,
    row_count: u64,
    col_count: u64,
) -> Result<TensorValuesPreview, String> {
    let row_count = row_count.min(128);
    let col_count = col_count.min(128);
    let value_count = row_count
        .checked_mul(col_count)
        .ok_or_else(|| "tensor preview window is too large".to_string())?;
    if value_count > TENSOR_VALUES_MAX_WINDOW {
        return Err(format!(
            "Tensor preview window is limited to {TENSOR_VALUES_MAX_WINDOW} values"
        ));
    }

    let path = match source.as_deref() {
        Some("mmproj") => {
            let guard = projector_state.0.lock().map_err(|e| e.to_string())?;
            match &*guard {
                Some(model) => model.path.clone(),
                None => return Err("No MMPROJ loaded".into()),
            }
        }
        None | Some("model") => {
            let guard = state.0.lock().map_err(|e| e.to_string())?;
            match &*guard {
                Some(model) => model.path.clone(),
                None => return Err("No model loaded".into()),
            }
        }
        Some(source) => return Err(format!("Unknown tensor source: {source}")),
    };
    let preview = runtime_bindings::preview_tensor_values(
        path.to_string_lossy().as_ref(),
        &tensor_name,
        row_offset,
        col_offset,
        row_count,
        col_count,
    )?;

    Ok(TensorValuesPreview {
        values: preview.values,
        rows: preview.rows,
        cols: preview.cols,
        total_rows: preview.total_rows,
        total_cols: preview.total_cols,
    })
}

fn parse_default_quant(value: &str) -> QuantType {
    match value {
        "F32" => QuantType::F32,
        "BF16" => QuantType::BF16,
        "F16" => QuantType::F16,
        "Q8_0" => QuantType::Q8_0,
        "Q6_K" => QuantType::Q6_K,
        "Q5_K" => QuantType::Q5_K,
        "Q5_K_M" => QuantType::Q5_K_M,
        "Q5_1" => QuantType::Q5_1,
        "Q5_0" => QuantType::Q5_0,
        "Q4_K" => QuantType::Q4_K,
        "Q4_K_M" => QuantType::Q4_K_M,
        "Q4_1" => QuantType::Q4_1,
        "Q4_0" => QuantType::Q4_0,
        "Q3_K" => QuantType::Q3_K,
        "Q3_K_M" => QuantType::Q3_K_M,
        "Q2_K" => QuantType::Q2_K,
        _ => QuantType::Q4_K_M,
    }
}

#[cfg(test)]
mod tests {
    use super::is_standalone_projector_gguf;

    #[test]
    fn rejects_standalone_projector_ggufs_but_keeps_integrated_models() {
        assert!(is_standalone_projector_gguf(
            "clip",
            ["v.patch_embd.weight", "mm.0.weight"].into_iter(),
        ));
        assert!(!is_standalone_projector_gguf(
            "llama",
            ["token_embd.weight", "v.patch_embd.weight"].into_iter(),
        ));
    }
}
