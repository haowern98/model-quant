use std::sync::Mutex;
use tauri::State;

use crate::quant::recipe::{QuantType, RecipeState};

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
) -> Result<RecipeState, String> {
    let mut guard = state.0.lock().map_err(|e| e.to_string())?;
    let recipe = guard.as_mut().ok_or("No recipe initialized")?;
    let qt = parse_quant_type(&quant_type)?;
    recipe.assign_all(qt);
    Ok(recipe.clone())
}

#[tauri::command]
pub async fn assign_by_pattern(
    pattern: String,
    quant_type: String,
    state: State<'_, RecipeStore>,
) -> Result<RecipeState, String> {
    let mut guard = state.0.lock().map_err(|e| e.to_string())?;
    let recipe = guard.as_mut().ok_or("No recipe initialized")?;
    let qt = parse_quant_type(&quant_type)?;
    recipe.assign_by_pattern(&pattern, qt);
    Ok(recipe.clone())
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
        "Q4_K" => Ok(QuantType::Q4_K),
        "Q4_K_M" => Ok(QuantType::Q4_K_M),
        "Q3_K" => Ok(QuantType::Q3_K),
        "Q3_K_M" => Ok(QuantType::Q3_K_M),
        "Q2_K" => Ok(QuantType::Q2_K),
        _ => Err(format!("Unknown quant type: {}", s)),
    }
}
