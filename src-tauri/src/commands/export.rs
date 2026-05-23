use std::path::PathBuf;
use tauri::State;

use crate::commands::model::ModelState;
use crate::commands::quant::RecipeStore;
use crate::profile::benchmark::{run_benchmark, BenchmarkResult};
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
    app: tauri::AppHandle,
    state: State<'_, RecipeStore>,
) -> Result<BenchmarkResult, String> {
    let progress = ProgressEmitter::new(app.clone());

    progress.requantizing(0.0, "starting...");
    let source = PathBuf::from(&recipe.base_model);
    let temp_path = std::env::temp_dir().join("model-surgery-temp.gguf");

    crate::quant::engine::apply_recipe_stub(&source, &temp_path, &recipe)
        .map_err(|e| e.to_string())?;
    progress.requantizing(1.0, "done");

    progress.writing(0.5, "temp.gguf");
    progress.writing(1.0, "done");

    let result = run_benchmark(&temp_path, prompt_tokens, &progress)?;

    let _ = std::fs::remove_file(&temp_path);

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
