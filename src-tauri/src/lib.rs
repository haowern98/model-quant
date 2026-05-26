pub mod commands;
pub mod ffi;
pub mod gguf;
pub mod profile;
pub mod progress;
pub mod quant;

use commands::model::ModelState;
use commands::quant::RecipeStore;
use std::sync::Mutex;

pub fn run() {
    // Init C++ profiler (CUDA or stub, depending on build)
    unsafe {
        crate::ffi::profiler_bindings::profiler_init();
    }

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
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
            commands::eval_backend::get_official_eval_backend_status,
            commands::eval_backend::install_official_eval_backend,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");

    unsafe {
        crate::ffi::profiler_bindings::profiler_shutdown();
    }
}
