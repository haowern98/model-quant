use std::sync::Mutex;
use tauri::State;

use crate::commands::model::ModelState;
use crate::quant::recipe::{AllowedTargetQuants, QuantType, RecipeState};

pub struct RecipeStore(pub Mutex<Option<RecipeState>>);

#[tauri::command]
pub async fn assign_quant(
    tensor_names: Vec<String>,
    quant_type: String,
    state: State<'_, RecipeStore>,
) -> Result<RecipeState, String> {
    let mut guard = state.0.lock().map_err(|e| e.to_string())?;
    let recipe = guard.as_mut().ok_or("No recipe initialized")?;
    let qt = parse_quant_type(&quant_type)?;
    recipe.assign_tensors(&tensor_names, qt);
    Ok(recipe.clone())
}

#[tauri::command]
pub async fn assign_all(
    quant_type: String,
    state: State<'_, RecipeStore>,
    model_state: State<'_, ModelState>,
) -> Result<RecipeState, String> {
    let qt = parse_quant_type(&quant_type)?;
    let allowed_target_quants = loaded_allowed_target_quants(&model_state)?;
    let mut guard = state.0.lock().map_err(|e| e.to_string())?;
    let recipe = guard.as_mut().ok_or("No recipe initialized")?;
    recipe.assign_all_with_preflight(qt, &allowed_target_quants);
    Ok(recipe.clone())
}

#[tauri::command]
pub async fn assign_by_pattern(
    pattern: String,
    quant_type: String,
    state: State<'_, RecipeStore>,
    model_state: State<'_, ModelState>,
) -> Result<RecipeState, String> {
    let qt = parse_quant_type(&quant_type)?;
    let allowed_target_quants = loaded_allowed_target_quants(&model_state)?;
    let mut guard = state.0.lock().map_err(|e| e.to_string())?;
    let recipe = guard.as_mut().ok_or("No recipe initialized")?;
    recipe.assign_by_pattern_with_preflight(&pattern, qt, &allowed_target_quants);
    Ok(recipe.clone())
}

fn loaded_allowed_target_quants(model_state: &ModelState) -> Result<AllowedTargetQuants, String> {
    let guard = model_state.0.lock().map_err(|e| e.to_string())?;
    let model = guard.as_ref().ok_or("No model loaded")?;

    Ok(model
        .info
        .tensors
        .iter()
        .map(|tensor| {
            (
                tensor.name.clone(),
                tensor.quant_preflight.allowed_target_quants.clone(),
            )
        })
        .collect())
}

fn parse_quant_type(s: &str) -> Result<QuantType, String> {
    match s {
        "F32" => Ok(QuantType::F32),
        "BF16" => Ok(QuantType::BF16),
        "F16" => Ok(QuantType::F16),
        "Q8_0" => Ok(QuantType::Q8_0),
        "Q6_K" => Ok(QuantType::Q6_K),
        "Q5_K" => Ok(QuantType::Q5_K),
        "Q5_K_M" => Ok(QuantType::Q5_K_M),
        "Q5_1" => Ok(QuantType::Q5_1),
        "Q5_0" => Ok(QuantType::Q5_0),
        "Q4_K" => Ok(QuantType::Q4_K),
        "Q4_K_M" => Ok(QuantType::Q4_K_M),
        "Q4_1" => Ok(QuantType::Q4_1),
        "Q4_0" => Ok(QuantType::Q4_0),
        "Q3_K" => Ok(QuantType::Q3_K),
        "Q3_K_M" => Ok(QuantType::Q3_K_M),
        "Q2_K" => Ok(QuantType::Q2_K),
        _ => Err(format!("Unknown quant type: {}", s)),
    }
}
