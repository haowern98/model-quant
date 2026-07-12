pub mod commands;
pub mod ffi;
pub mod gguf;
pub mod profile;
pub mod progress;
pub mod quant;

use commands::hardware::HardwareMonitor;
use commands::model::{ModelState, ProjectorState};
use commands::modelinspector_api::ModelInspectorApiState;
use commands::official_benchmarks::OfficialBenchmarkRunner;
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
        .manage(ProjectorState(Mutex::new(None)))
        .manage(RecipeStore(Mutex::new(None)))
        .manage(HardwareMonitor::new())
        .manage(ModelInspectorApiState::new())
        .manage(OfficialBenchmarkRunner::new())
        .invoke_handler(tauri::generate_handler![
            commands::model::open_model,
            commands::model::open_projector,
            commands::model::close_projector,
            commands::model::get_tensors,
            commands::model::get_tensor_values,
            commands::quant::assign_quant,
            commands::quant::assign_all,
            commands::quant::assign_by_pattern,
            commands::export::save_recipe,
            commands::export::export_gguf,
            commands::export::load_recipe,
            commands::export::list_recipes,
            commands::export::test_recipe,
            commands::export::cancel_recipe_test,
            commands::hardware::get_hardware_snapshot,
            commands::modelinspector_api::start_modelinspector_api,
            commands::modelinspector_api::stop_modelinspector_api,
            commands::modelinspector_api::get_modelinspector_api_status,
            commands::official_benchmarks::get_gpqa_diamond_status,
            commands::official_benchmarks::get_humaneval_status,
            commands::official_benchmarks::get_terminal_bench_status,
            commands::official_benchmarks::get_humaneval_dataset_status,
            commands::official_benchmarks::get_terminal_bench_dataset_status,
            commands::official_benchmarks::get_gpqa_diamond_dataset_rows,
            commands::official_benchmarks::get_humaneval_dataset_rows,
            commands::official_benchmarks::get_terminal_bench_dataset_rows,
            commands::official_benchmarks::install_gpqa_diamond_harness,
            commands::official_benchmarks::install_humaneval_harness,
            commands::official_benchmarks::download_gpqa_diamond_dataset,
            commands::official_benchmarks::download_humaneval_dataset,
            commands::official_benchmarks::download_terminal_bench_dataset,
            commands::official_benchmarks::delete_gpqa_diamond_dataset,
            commands::official_benchmarks::delete_humaneval_dataset,
            commands::official_benchmarks::delete_terminal_bench_dataset,
            commands::official_benchmarks::delete_gpqa_diamond_harness,
            commands::official_benchmarks::delete_humaneval_harness,
            commands::official_benchmarks::run_gpqa_diamond_benchmark,
            commands::official_benchmarks::run_humaneval_benchmark,
            commands::official_benchmarks::run_terminal_bench_benchmark,
            commands::official_benchmarks::cancel_official_benchmark,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");

    unsafe {
        crate::ffi::profiler_bindings::profiler_shutdown();
    }
}
