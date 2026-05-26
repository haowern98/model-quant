use std::path::PathBuf;
use tauri::State;

use crate::commands::model::ModelState;
use crate::commands::quant::RecipeStore;
use crate::profile::benchmark::{
    run_native_recipe_compare_benchmark, run_native_recipe_single_benchmark, BenchmarkResult,
};
use crate::progress::ProgressEmitter;
use crate::quant::recipe::{RecipeState, RecipeStatus};

#[tauri::command]
pub async fn save_recipe(path: String, recipe: RecipeState) -> Result<(), String> {
    let json = serde_json::to_string_pretty(&recipe).map_err(|e| e.to_string())?;
    std::fs::write(&path, json).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn export_gguf(path: String, recipe: RecipeState) -> Result<(), String> {
    let source = PathBuf::from(&recipe.base_model);
    if !source.exists() {
        return Err(format!("Source model not found: {}", recipe.base_model));
    }

    let dest = PathBuf::from(&path);
    crate::quant::engine::apply_recipe_stub(&source, &dest, &recipe).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn load_recipe(
    path: String,
    state: State<'_, RecipeStore>,
    model_state: State<'_, ModelState>,
) -> Result<RecipeState, String> {
    let json = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
    let recipe: RecipeState = serde_json::from_str(&json).map_err(|e| e.to_string())?;

    {
        let model_guard = model_state.0.lock().map_err(|e| e.to_string())?;
        let model = model_guard
            .as_ref()
            .ok_or("Open a GGUF model before loading a recipe")?;
        let current_tensors = model
            .tensors
            .iter()
            .map(|t| t.name.as_str())
            .collect::<Vec<_>>();
        let recipe_tensors = recipe
            .assignments
            .iter()
            .map(|a| a.tensor_name.as_str())
            .collect::<Vec<_>>();
        validate_recipe_tensors(&current_tensors, &recipe_tensors)?;
    }

    let mut guard = state.0.lock().map_err(|e| e.to_string())?;
    *guard = Some(recipe.clone());
    Ok(recipe)
}

fn validate_recipe_tensors(
    current_tensors: &[&str],
    recipe_tensors: &[&str],
) -> Result<(), String> {
    let mut current = current_tensors.to_vec();
    let mut recipe = recipe_tensors.to_vec();
    current.sort_unstable();
    recipe.sort_unstable();

    if current == recipe {
        return Ok(());
    }

    let missing = current
        .iter()
        .filter(|name| recipe.binary_search(name).is_err())
        .copied()
        .take(5)
        .collect::<Vec<_>>();
    let extra = recipe
        .iter()
        .filter(|name| current.binary_search(name).is_err())
        .copied()
        .take(5)
        .collect::<Vec<_>>();

    let duplicate_count = recipe.windows(2).filter(|pair| pair[0] == pair[1]).count();

    let mut details = Vec::new();
    if !missing.is_empty() {
        details.push(format!("missing from recipe: {}", missing.join(", ")));
    }
    if !extra.is_empty() {
        details.push(format!("not in current model: {}", extra.join(", ")));
    }
    if duplicate_count > 0 {
        details.push(format!(
            "duplicate recipe tensor entries: {}",
            duplicate_count
        ));
    }

    if details.is_empty() {
        details.push(format!(
            "current model has {} tensors, recipe has {} assignments",
            current.len(),
            recipe.len()
        ));
    }

    Err(format!(
        "Recipe does not match the currently opened model ({})",
        details.join("; ")
    ))
}

#[cfg(test)]
mod tests {
    use super::validate_recipe_tensors;

    #[test]
    fn accepts_exact_tensor_set_in_any_order() {
        let current = [
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "output.weight",
        ];
        let recipe = [
            "output.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_q.weight",
        ];

        assert!(validate_recipe_tensors(&current, &recipe).is_ok());
    }

    #[test]
    fn rejects_recipe_missing_current_tensor() {
        let current = ["blk.0.attn_q.weight", "blk.0.attn_k.weight"];
        let recipe = ["blk.0.attn_q.weight"];

        let err = validate_recipe_tensors(&current, &recipe).unwrap_err();
        assert!(err.contains("missing from recipe: blk.0.attn_k.weight"));
    }

    #[test]
    fn rejects_recipe_with_extra_tensor() {
        let current = ["blk.0.attn_q.weight"];
        let recipe = ["blk.0.attn_q.weight", "blk.99.attn_q.weight"];

        let err = validate_recipe_tensors(&current, &recipe).unwrap_err();
        assert!(err.contains("not in current model: blk.99.attn_q.weight"));
    }

    #[test]
    fn rejects_duplicate_recipe_tensor_entries() {
        let current = ["blk.0.attn_q.weight"];
        let recipe = ["blk.0.attn_q.weight", "blk.0.attn_q.weight"];

        let err = validate_recipe_tensors(&current, &recipe).unwrap_err();
        assert!(err.contains("duplicate recipe tensor entries: 1"));
    }
}

#[tauri::command]
pub async fn list_recipes() -> Result<Vec<RecipeState>, String> {
    let mut recipes = Vec::new();
    if let Ok(entries) = std::fs::read_dir(".") {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "json") {
                if let Ok(json) = std::fs::read_to_string(&path) {
                    if let Ok(recipe) = serde_json::from_str::<RecipeState>(&json) {
                        recipes.push(recipe);
                    }
                }
            }
        }
    }
    Ok(recipes)
}

#[tauri::command]
pub async fn test_recipe(
    recipe: RecipeState,
    prompt_tokens: u32,
    test_mode: Option<String>,
    app: tauri::AppHandle,
    state: State<'_, RecipeStore>,
) -> Result<BenchmarkResult, String> {
    let progress = ProgressEmitter::new(app.clone());

    let source = PathBuf::from(&recipe.base_model);
    if !source.exists() {
        return Err(format!("Source model not found: {}", recipe.base_model));
    }

    let targets = recipe_targets(&recipe);
    let analysis =
        crate::ffi::runtime_bindings::analyze_recipe(&source.to_string_lossy(), &targets)?;
    validate_recipe_analysis(&analysis)?;

    progress.requantizing(1.0, "skipped");
    progress.writing(1.0, "no temporary GGUF written");
    let result = match test_mode.as_deref().unwrap_or("compare_baseline") {
        "single" => {
            run_native_recipe_single_benchmark(&source, &targets, prompt_tokens, &progress)?
        }
        "compare_baseline" => {
            run_native_recipe_compare_benchmark(&source, &targets, prompt_tokens, &progress)?
        }
        mode => return Err(format!("Unknown test recipe mode: {}", mode)),
    };

    {
        let mut guard = state.0.lock().map_err(|e| e.to_string())?;
        if let Some(recipe) = guard.as_mut() {
            recipe.status = RecipeStatus::Profiled;
            recipe.profile = Some(crate::quant::recipe::RecipeProfile {
                vram_estimate: result.vram_allocated_mb,
                size_saved_vs_q8: 0.0,
            });
        }
    }

    Ok(result)
}

fn recipe_targets(recipe: &RecipeState) -> Vec<(String, String)> {
    recipe
        .assignments
        .iter()
        .map(|assignment| {
            (
                assignment.tensor_name.clone(),
                quant_type_name(&assignment.quant_type).to_string(),
            )
        })
        .collect::<Vec<_>>()
}

fn validate_recipe_analysis(
    analysis: &crate::ffi::runtime_bindings::MsRecipeAnalysis,
) -> Result<(), String> {
    if analysis.missing_count > 0 || analysis.unknown_quant_count > 0 {
        return Err(format!(
            "Recipe preflight failed: {} missing tensor(s), {} unknown quant target(s).",
            analysis.missing_count, analysis.unknown_quant_count
        ));
    }

    if analysis.unsupported_count > 0 {
        return Err(format!(
            "Recipe preflight found {} unsupported tensor conversion(s). Phase 2 supports only F32/F16/BF16 to Q8_0.",
            analysis.unsupported_count
        ));
    }

    Ok(())
}

fn quant_type_name(quant_type: &crate::quant::recipe::QuantType) -> &'static str {
    match quant_type {
        crate::quant::recipe::QuantType::F32 => "F32",
        crate::quant::recipe::QuantType::BF16 => "BF16",
        crate::quant::recipe::QuantType::F16 => "F16",
        crate::quant::recipe::QuantType::Q8_0 => "Q8_0",
        crate::quant::recipe::QuantType::Q6_K => "Q6_K",
        crate::quant::recipe::QuantType::Q5_K => "Q5_K",
        crate::quant::recipe::QuantType::Q5_K_M => "Q5_K_M",
        crate::quant::recipe::QuantType::Q4_K => "Q4_K",
        crate::quant::recipe::QuantType::Q4_K_M => "Q4_K_M",
        crate::quant::recipe::QuantType::Q3_K => "Q3_K",
        crate::quant::recipe::QuantType::Q3_K_M => "Q3_K_M",
        crate::quant::recipe::QuantType::Q2_K => "Q2_K",
    }
}
