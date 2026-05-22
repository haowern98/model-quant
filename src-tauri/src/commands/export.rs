use std::path::PathBuf;
use tauri::State;

use crate::quant::recipe::{RecipeState, RecipeStatus};
use crate::commands::quant::RecipeStore;
use crate::profile::benchmark::{run_benchmark, BenchmarkResult};
use crate::progress::ProgressEmitter;

#[tauri::command]
pub async fn save_recipe(
    path: String,
    recipe: RecipeState,
) -> Result<(), String> {
    let json = serde_json::to_string_pretty(&recipe).map_err(|e| e.to_string())?;
    std::fs::write(&path, json).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn export_gguf(
    path: String,
    recipe: RecipeState,
) -> Result<(), String> {
    let source = PathBuf::from(&recipe.base_model);
    if !source.exists() {
        return Err(format!("Source model not found: {}", recipe.base_model));
    }

    let dest = PathBuf::from(&path);
    crate::quant::engine::apply_recipe_stub(&source, &dest, &recipe)
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn load_recipe(
    path: String,
    state: State<'_, RecipeStore>,
) -> Result<RecipeState, String> {
    let json = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
    let recipe: RecipeState = serde_json::from_str(&json).map_err(|e| e.to_string())?;
    let mut guard = state.0.lock().map_err(|e| e.to_string())?;
    *guard = Some(recipe.clone());
    Ok(recipe)
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
