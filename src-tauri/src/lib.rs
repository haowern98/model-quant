pub mod gguf;
pub mod ffi;
pub mod quant;
pub mod progress;
pub mod profile;
pub mod commands;

use std::sync::Mutex;
use commands::model::ModelState;
use commands::quant::RecipeStore;

pub fn run() {
    tauri::Builder::default()
        .manage(ModelState(Mutex::new(None)))
        .manage(RecipeStore(Mutex::new(None)))
        .invoke_handler(tauri::generate_handler![
            commands::model::open_model,
            commands::model::get_tensors,
            commands::quant::assign_quant,
            commands::quant::assign_all,
            commands::quant::assign_by_pattern,
            commands::export::save_recipe,
            commands::export::export_gguf,
            commands::export::load_recipe,
            commands::export::list_recipes,
            commands::export::test_recipe,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
