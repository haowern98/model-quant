use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::json;
use tauri::{Manager, State};

use crate::commands::modelinspector_api::{
    modelinspector_api_model_summary, modelinspector_api_runtime_totals,
    modelinspector_api_tensor_summary, ModelInspectorApiModelSummary,
    ModelInspectorApiRuntimeSummary, ModelInspectorApiRuntimeTotals, ModelInspectorApiState,
    ModelInspectorApiTensorSummary,
};
use crate::profile::benchmark::{BenchmarkResult, StandardEvalReport, StandardEvalTaskReport};

const GPQA_SAMPLE_COUNT: u64 = 198;
const GPQA_DATASET_ID: &str = "AI-ModelScope/gpqa_diamond";
const GPQA_DATASET_MARKER_VERSION: u32 = 1;
const HUMANEVAL_SAMPLE_COUNT: u64 = 164;
const HUMANEVAL_DATASET_ID: &str = "opencompass/humaneval";
const HUMANEVAL_DATASET_MARKER_VERSION: u32 = 1;
const EVALSCOPE_VERSION: &str = "1.8.0";
const GPQA_DEFAULT_CONTEXT_WINDOW: u32 = 20_000;
const GPQA_DEFAULT_TEMPERATURE: f64 = 0.0;
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x08000000;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GpqaDiamondStatus {
    pub ready: bool,
    pub status_label: String,
    pub python: Option<String>,
    pub evalscope: Option<String>,
    pub dataset_ready: bool,
    pub dataset_status_label: String,
    pub dataset_path: Option<String>,
    pub dataset_hash: Option<String>,
    pub dataset_url: String,
    pub expected_dataset_hash: String,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HumanEvalStatus {
    pub ready: bool,
    pub status_label: String,
    pub python: Option<String>,
    pub evalscope: Option<String>,
    pub docker_ready: bool,
    pub docker: Option<String>,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HumanEvalDatasetStatus {
    pub dataset_ready: bool,
    pub dataset_status_label: String,
    pub dataset_path: Option<String>,
    pub dataset_hash: Option<String>,
    pub dataset_url: String,
    pub expected_dataset_hash: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct GpqaDatasetRow {
    pub index: usize,
    pub question: String,
    pub choices: Vec<String>,
    pub answer: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct HumanEvalDatasetRow {
    pub index: usize,
    pub task_id: String,
    pub entry_point: String,
    pub prompt: String,
    pub canonical_solution: String,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GpqaShotMode {
    ZeroShot,
    FiveShotCot,
}

impl GpqaShotMode {
    fn few_shot_num(self) -> u8 {
        match self {
            Self::ZeroShot => 0,
            Self::FiveShotCot => 5,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::ZeroShot => "0-shot CoT",
            Self::FiveShotCot => "5-shot CoT",
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GpqaRunConfig {
    pub context_window: Option<u32>,
    pub sample_limit: Option<u64>,
    pub temperature: Option<f64>,
    pub thinking: Option<GpqaThinkingMode>,
    pub top_k: Option<i32>,
    pub repeat_penalty: Option<f64>,
    pub presence_penalty: Option<f64>,
    pub top_p: Option<f64>,
    pub min_p: Option<f64>,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum GpqaThinkingMode {
    Off,
    On,
}

impl GpqaThinkingMode {
    fn enable_thinking(self) -> bool {
        matches!(self, Self::On)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct EffectiveGpqaRunConfig {
    context_window: u32,
    sample_limit: u64,
    temperature: f64,
    thinking: GpqaThinkingMode,
    top_k: Option<i32>,
    repeat_penalty: Option<f64>,
    presence_penalty: Option<f64>,
    top_p: Option<f64>,
    min_p: Option<f64>,
}

fn effective_gpqa_run_config(
    config: Option<GpqaRunConfig>,
) -> Result<EffectiveGpqaRunConfig, String> {
    let config = config.unwrap_or(GpqaRunConfig {
        context_window: None,
        sample_limit: None,
        temperature: None,
        thinking: None,
        top_k: None,
        repeat_penalty: None,
        presence_penalty: None,
        top_p: None,
        min_p: None,
    });
    let context_window = config.context_window.unwrap_or(GPQA_DEFAULT_CONTEXT_WINDOW);
    if context_window == 0 {
        return Err(format!("GPQA context window must be greater than 0."));
    }

    let sample_limit = config.sample_limit.unwrap_or(GPQA_SAMPLE_COUNT);
    if sample_limit == 0 || sample_limit > GPQA_SAMPLE_COUNT {
        return Err(format!(
            "GPQA sample limit must be between 1 and {GPQA_SAMPLE_COUNT}."
        ));
    }

    let temperature = config.temperature.unwrap_or(GPQA_DEFAULT_TEMPERATURE);
    if !temperature.is_finite() || !(0.0..=2.0).contains(&temperature) {
        return Err("GPQA temperature must be between 0 and 2.".to_string());
    }
    let thinking = config.thinking.unwrap_or(GpqaThinkingMode::Off);
    if let Some(top_k) = config.top_k {
        if !(0..=1000).contains(&top_k) {
            return Err("GPQA top K sampling must be between 0 and 1000.".to_string());
        }
    }
    if let Some(repeat_penalty) = config.repeat_penalty {
        if !repeat_penalty.is_finite() || !(0.0..=3.0).contains(&repeat_penalty) {
            return Err("GPQA repeat penalty must be between 0 and 3.".to_string());
        }
    }
    if let Some(presence_penalty) = config.presence_penalty {
        if !presence_penalty.is_finite() || !(-2.0..=2.0).contains(&presence_penalty) {
            return Err("GPQA presence penalty must be between -2 and 2.".to_string());
        }
    }
    if let Some(top_p) = config.top_p {
        if !top_p.is_finite() || !(0.0..=1.0).contains(&top_p) {
            return Err("GPQA top P sampling must be between 0 and 1.".to_string());
        }
    }
    if let Some(min_p) = config.min_p {
        if !min_p.is_finite() || !(0.0..=1.0).contains(&min_p) {
            return Err("GPQA min P sampling must be between 0 and 1.".to_string());
        }
    }

    Ok(EffectiveGpqaRunConfig {
        context_window,
        sample_limit,
        temperature,
        thinking,
        top_k: config.top_k,
        repeat_penalty: config.repeat_penalty,
        presence_penalty: config.presence_penalty,
        top_p: config.top_p,
        min_p: config.min_p,
    })
}

fn gpqa_generation_config(effective_config: &EffectiveGpqaRunConfig) -> serde_json::Value {
    let mut generation_config = json!({
        "temperature": effective_config.temperature,
        "stream": false,
        "chat_template_kwargs": {
            "enable_thinking": effective_config.thinking.enable_thinking()
        }
    });
    if let Some(top_k) = effective_config.top_k {
        generation_config["top_k"] = json!(top_k);
    }
    if let Some(repeat_penalty) = effective_config.repeat_penalty {
        generation_config["repeat_penalty"] = json!(repeat_penalty);
    }
    if let Some(presence_penalty) = effective_config.presence_penalty {
        generation_config["presence_penalty"] = json!(presence_penalty);
    }
    if let Some(top_p) = effective_config.top_p {
        generation_config["top_p"] = json!(top_p);
    }
    if let Some(min_p) = effective_config.min_p {
        generation_config["min_p"] = json!(min_p);
    }
    generation_config
}

#[derive(Clone)]
pub struct OfficialBenchmarkRunner {
    child: Arc<Mutex<Option<Child>>>,
}

impl OfficialBenchmarkRunner {
    pub fn new() -> Self {
        Self {
            child: Arc::new(Mutex::new(None)),
        }
    }
}

#[derive(Debug, Clone)]
struct PythonCommand {
    executable: String,
    prefix_args: Vec<String>,
}

#[derive(Debug, Clone)]
enum GpqaDatasetState {
    Missing,
    Verified {
        path: PathBuf,
        hash: String,
    },
    Invalid {
        path: PathBuf,
        hash: Option<String>,
        detail: String,
    },
}

#[tauri::command]
pub async fn get_gpqa_diamond_status(app: tauri::AppHandle) -> Result<GpqaDiamondStatus, String> {
    let app_data_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to resolve app data directory: {e}"))?;
    tauri::async_runtime::spawn_blocking(move || detect_gpqa_diamond_status(app_data_dir))
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn get_humaneval_status(app: tauri::AppHandle) -> Result<HumanEvalStatus, String> {
    let app_data_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to resolve app data directory: {e}"))?;
    tauri::async_runtime::spawn_blocking(move || detect_humaneval_status(app_data_dir))
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn get_humaneval_dataset_status(
    app: tauri::AppHandle,
) -> Result<HumanEvalDatasetStatus, String> {
    let app_data_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to resolve app data directory: {e}"))?;
    Ok(detect_humaneval_dataset_status(&app_data_dir))
}

#[tauri::command]
pub async fn install_gpqa_diamond_harness(
    app: tauri::AppHandle,
    runner: State<'_, OfficialBenchmarkRunner>,
) -> Result<GpqaDiamondStatus, String> {
    let app_data_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to resolve app data directory: {e}"))?;
    let child = runner.child.clone();
    tauri::async_runtime::spawn_blocking(move || {
        install_gpqa_diamond_harness_blocking(app_data_dir, app, child)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
pub async fn install_humaneval_harness(
    app: tauri::AppHandle,
    runner: State<'_, OfficialBenchmarkRunner>,
) -> Result<HumanEvalStatus, String> {
    let app_data_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to resolve app data directory: {e}"))?;
    let child = runner.child.clone();
    tauri::async_runtime::spawn_blocking(move || {
        install_gpqa_diamond_harness_blocking(app_data_dir.clone(), app, child)?;
        Ok(detect_humaneval_status(app_data_dir))
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
pub async fn download_gpqa_diamond_dataset(
    app: tauri::AppHandle,
    runner: State<'_, OfficialBenchmarkRunner>,
) -> Result<GpqaDiamondStatus, String> {
    let app_data_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to resolve app data directory: {e}"))?;
    let child = runner.child.clone();
    tauri::async_runtime::spawn_blocking(move || {
        download_gpqa_diamond_dataset_blocking(app_data_dir, app, child)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
pub async fn download_humaneval_dataset(
    app: tauri::AppHandle,
    runner: State<'_, OfficialBenchmarkRunner>,
) -> Result<HumanEvalDatasetStatus, String> {
    let app_data_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to resolve app data directory: {e}"))?;
    let child = runner.child.clone();
    tauri::async_runtime::spawn_blocking(move || {
        download_humaneval_dataset_blocking(app_data_dir, app, child)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
pub async fn delete_gpqa_diamond_dataset(
    app: tauri::AppHandle,
    runner: State<'_, OfficialBenchmarkRunner>,
) -> Result<GpqaDiamondStatus, String> {
    let app_data_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to resolve app data directory: {e}"))?;
    let child = runner.child.clone();
    tauri::async_runtime::spawn_blocking(move || {
        ensure_official_benchmark_idle(&child)?;
        remove_path_if_exists(&gpqa_dataset_cache_root(&app_data_dir))?;
        Ok(detect_gpqa_diamond_status(app_data_dir))
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
pub async fn delete_humaneval_dataset(
    app: tauri::AppHandle,
    runner: State<'_, OfficialBenchmarkRunner>,
) -> Result<HumanEvalDatasetStatus, String> {
    let app_data_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to resolve app data directory: {e}"))?;
    let child = runner.child.clone();
    tauri::async_runtime::spawn_blocking(move || {
        ensure_official_benchmark_idle(&child)?;
        remove_path_if_exists(&humaneval_dataset_cache_root(&app_data_dir))?;
        Ok(detect_humaneval_dataset_status(&app_data_dir))
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
pub async fn delete_gpqa_diamond_harness(
    app: tauri::AppHandle,
    runner: State<'_, OfficialBenchmarkRunner>,
) -> Result<GpqaDiamondStatus, String> {
    let app_data_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to resolve app data directory: {e}"))?;
    let child = runner.child.clone();
    tauri::async_runtime::spawn_blocking(move || {
        ensure_official_benchmark_idle(&child)?;
        remove_path_if_exists(&gpqa_env_dir(&app_data_dir).join("venv"))?;
        Ok(detect_gpqa_diamond_status(app_data_dir))
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
pub async fn delete_humaneval_harness(
    app: tauri::AppHandle,
    runner: State<'_, OfficialBenchmarkRunner>,
) -> Result<HumanEvalStatus, String> {
    let app_data_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to resolve app data directory: {e}"))?;
    let child = runner.child.clone();
    tauri::async_runtime::spawn_blocking(move || {
        ensure_official_benchmark_idle(&child)?;
        run_managed_child(
            &venv_python_path(&gpqa_env_dir(&app_data_dir)).to_string_lossy(),
            vec![
                "-m".to_string(),
                "pip".to_string(),
                "uninstall".to_string(),
                "-y".to_string(),
                "ms-enclave".to_string(),
            ],
            &child,
        )?;
        Ok(detect_humaneval_status(app_data_dir))
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
pub async fn get_gpqa_diamond_dataset_rows(
    app: tauri::AppHandle,
) -> Result<Vec<GpqaDatasetRow>, String> {
    let app_data_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to resolve app data directory: {e}"))?;
    tauri::async_runtime::spawn_blocking(move || read_gpqa_dataset_rows(&app_data_dir))
        .await
        .map_err(|e| e.to_string())?
}

#[tauri::command]
pub async fn get_humaneval_dataset_rows(
    app: tauri::AppHandle,
) -> Result<Vec<HumanEvalDatasetRow>, String> {
    let app_data_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to resolve app data directory: {e}"))?;
    tauri::async_runtime::spawn_blocking(move || read_humaneval_dataset_rows(&app_data_dir))
        .await
        .map_err(|e| e.to_string())?
}

#[tauri::command]
pub async fn run_gpqa_diamond_benchmark(
    base_url: String,
    api_key: String,
    model_id: String,
    shot_mode: GpqaShotMode,
    config: Option<GpqaRunConfig>,
    app: tauri::AppHandle,
    api_state: State<'_, ModelInspectorApiState>,
    runner: State<'_, OfficialBenchmarkRunner>,
) -> Result<BenchmarkResult, String> {
    let tensor_summary = modelinspector_api_tensor_summary(&api_state, &base_url, &model_id)?;
    let model_summary = modelinspector_api_model_summary(&api_state, &base_url, &model_id)?;
    let runtime_totals = modelinspector_api_runtime_totals(&api_state, &base_url, &model_id)?;
    let child = runner.child.clone();
    tauri::async_runtime::spawn_blocking(move || {
        run_gpqa_diamond_blocking(
            base_url,
            api_key,
            model_id,
            shot_mode,
            config,
            tensor_summary,
            model_summary,
            runtime_totals,
            app,
            child,
        )
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
pub async fn run_humaneval_benchmark(
    base_url: String,
    api_key: String,
    model_id: String,
    config: Option<GpqaRunConfig>,
    app: tauri::AppHandle,
    api_state: State<'_, ModelInspectorApiState>,
    runner: State<'_, OfficialBenchmarkRunner>,
) -> Result<BenchmarkResult, String> {
    let tensor_summary = modelinspector_api_tensor_summary(&api_state, &base_url, &model_id)?;
    let model_summary = modelinspector_api_model_summary(&api_state, &base_url, &model_id)?;
    let runtime_totals = modelinspector_api_runtime_totals(&api_state, &base_url, &model_id)?;
    let child = runner.child.clone();
    tauri::async_runtime::spawn_blocking(move || {
        run_humaneval_blocking(
            base_url,
            api_key,
            model_id,
            config,
            tensor_summary,
            model_summary,
            runtime_totals,
            app,
            child,
        )
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
pub fn cancel_official_benchmark(runner: State<'_, OfficialBenchmarkRunner>) {
    if let Ok(mut guard) = runner.child.lock() {
        if let Some(child) = guard.as_mut() {
            let _ = child.kill();
        }
    }
}

fn ensure_official_benchmark_idle(child_slot: &Arc<Mutex<Option<Child>>>) -> Result<(), String> {
    if child_slot.lock().map_err(|e| e.to_string())?.is_some() {
        Err("A benchmark setup or run is already active.".to_string())
    } else {
        Ok(())
    }
}

fn remove_path_if_exists(path: &Path) -> Result<(), String> {
    if !path.exists() {
        return Ok(());
    }
    if path.is_dir() {
        std::fs::remove_dir_all(path)
    } else {
        std::fs::remove_file(path)
    }
    .map_err(|e| format!("Failed to delete {}: {e}", path.display()))
}

fn detect_gpqa_diamond_status(app_data_dir: PathBuf) -> GpqaDiamondStatus {
    let env_dir = gpqa_env_dir(&app_data_dir);
    let venv_python = venv_python_path(&env_dir);
    let system_python_exists = find_python().is_some();
    let dataset_state = detect_gpqa_dataset_state(&app_data_dir);

    if !venv_python.exists() {
        return managed_env_status(false, system_python_exists, None, dataset_state);
    }

    let output = run_python_probe(&PythonCommand {
        executable: venv_python.to_string_lossy().to_string(),
        prefix_args: Vec::new(),
    });
    match output {
        Ok(probe) => managed_env_status(true, system_python_exists, Some(&probe), dataset_state),
        Err(error) => GpqaDiamondStatus {
            ready: false,
            status_label: "Needs harness".to_string(),
            python: None,
            evalscope: None,
            dataset_ready: matches!(dataset_state, GpqaDatasetState::Verified { .. }),
            dataset_status_label: dataset_status_label(&dataset_state).to_string(),
            dataset_path: dataset_path_string(&dataset_state),
            dataset_hash: dataset_hash_string(&dataset_state),
            dataset_url: GPQA_DATASET_ID.to_string(),
            expected_dataset_hash: "EvalScope dataset cache marker".to_string(),
            detail: error,
        },
    }
}

fn detect_humaneval_status(app_data_dir: PathBuf) -> HumanEvalStatus {
    let (docker_ready, docker, docker_detail) = detect_docker_status();
    let env_dir = gpqa_env_dir(&app_data_dir);
    let venv_python = venv_python_path(&env_dir);
    let system_python_exists = find_python().is_some();

    if !venv_python.exists() {
        return HumanEvalStatus {
            ready: false,
            status_label: if system_python_exists {
                "Needs harness"
            } else if !docker_ready {
                "Needs Docker"
            } else {
                "Needs Python"
            }
            .to_string(),
            python: None,
            evalscope: None,
            docker_ready,
            docker,
            detail: if system_python_exists {
                "HumanEval harness has not been installed in the managed benchmark environment."
                    .to_string()
            } else if !docker_ready {
                docker_detail
            } else {
                "Python was not found on PATH.".to_string()
            },
        };
    }

    let output = run_humaneval_probe(&PythonCommand {
        executable: venv_python.to_string_lossy().to_string(),
        prefix_args: Vec::new(),
    });
    match output {
        Ok(probe) => classify_humaneval_status(&probe, docker_ready, docker, docker_detail),
        Err(error) => HumanEvalStatus {
            ready: false,
            status_label: if docker_ready {
                "Needs harness"
            } else {
                "Needs Docker"
            }
            .to_string(),
            python: None,
            evalscope: None,
            docker_ready,
            docker,
            detail: if docker_ready { error } else { docker_detail },
        },
    }
}

fn install_gpqa_diamond_harness_blocking(
    app_data_dir: PathBuf,
    app: tauri::AppHandle,
    child_slot: Arc<Mutex<Option<Child>>>,
) -> Result<GpqaDiamondStatus, String> {
    let Some(system_python) = find_python() else {
        return Err("Python was not found on PATH.".to_string());
    };

    let progress = crate::progress::ProgressEmitter::new(app);
    let env_dir = gpqa_env_dir(&app_data_dir);
    let venv_dir = env_dir.join("venv");
    let venv_python = venv_python_path(&env_dir);
    std::fs::create_dir_all(&env_dir).map_err(|e| e.to_string())?;

    if !venv_python.exists() {
        progress.emit(
            crate::progress::ProgressStage::Benchmarking,
            0.1,
            "Creating GPQA benchmark Python environment...",
        );
        run_managed_child(
            &system_python.executable,
            system_python
                .prefix_args
                .iter()
                .cloned()
                .chain([
                    "-m".to_string(),
                    "venv".to_string(),
                    venv_dir.to_string_lossy().to_string(),
                ])
                .collect::<Vec<_>>(),
            &child_slot,
        )?;
    }

    progress.emit(
        crate::progress::ProgressStage::Benchmarking,
        0.4,
        "Updating benchmark environment package installer...",
    );
    run_managed_child(
        &venv_python.to_string_lossy(),
        ["-m", "pip", "install", "--upgrade", "pip"]
            .into_iter()
            .map(ToOwned::to_owned)
            .collect(),
        &child_slot,
    )?;

    progress.emit(
        crate::progress::ProgressStage::Benchmarking,
        0.7,
        "Installing EvalScope GPQA benchmark harness...",
    );
    run_managed_child(
        &venv_python.to_string_lossy(),
        vec![
            "-m".to_string(),
            "pip".to_string(),
            "install".to_string(),
            "--upgrade".to_string(),
            format!("evalscope[sandbox]=={EVALSCOPE_VERSION}"),
        ],
        &child_slot,
    )?;

    let probe = run_python_probe(&PythonCommand {
        executable: venv_python.to_string_lossy().to_string(),
        prefix_args: Vec::new(),
    })?;
    let status = managed_env_status(
        true,
        true,
        Some(&probe),
        detect_gpqa_dataset_state(&app_data_dir),
    );
    if status.status_label == "Needs harness" || status.status_label == "Needs Python" {
        return Err(status.detail);
    }
    Ok(status)
}

fn download_gpqa_diamond_dataset_blocking(
    app_data_dir: PathBuf,
    app: tauri::AppHandle,
    child_slot: Arc<Mutex<Option<Child>>>,
) -> Result<GpqaDiamondStatus, String> {
    let env_dir = gpqa_env_dir(&app_data_dir);
    let venv_python = venv_python_path(&env_dir);
    if !venv_python.exists() {
        return Err(
            "GPQA Diamond harness is not installed. Install the harness before downloading the dataset."
                .to_string(),
        );
    }

    let progress = crate::progress::ProgressEmitter::new(app);
    let dataset_root = gpqa_dataset_cache_root(&app_data_dir);
    std::fs::create_dir_all(&dataset_root).map_err(|e| e.to_string())?;

    progress.emit(
        crate::progress::ProgressStage::Benchmarking,
        0.2,
        "Materializing GPQA Diamond through EvalScope...",
    );

    let script = r#"
import json
import sys
from pathlib import Path

dataset_root, marker_path, rows_path = sys.argv[1], sys.argv[2], sys.argv[3]
from evalscope.config import TaskConfig
from evalscope.benchmarks.gpqa.gpqa_adapter import GPQAAdapter
from evalscope.api.benchmark import BenchmarkMeta
from evalscope.utils.multi_choices import MultipleChoiceTemplate

meta = BenchmarkMeta(
    name="gpqa_diamond",
    dataset_id="AI-ModelScope/gpqa_diamond",
    data_adapter=GPQAAdapter,
    subset_list=["default"],
    default_subset="default",
    eval_split="train",
    few_shot_num=5,
    prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
)

task_config = TaskConfig(
    model="modelinspector-dataset-check",
    eval_type="mock_llm",
    datasets=["gpqa_diamond"],
    dataset_dir=dataset_root,
    dataset_args={"gpqa_diamond": {"few_shot_num": 5}},
)

adapter = GPQAAdapter(benchmark_meta=meta, task_config=task_config)
dataset = adapter.load_dataset()
sample_count = sum(len(samples) for samples in dataset.values())
if sample_count != 198:
    raise SystemExit(f"Expected 198 GPQA Diamond samples, got {sample_count}")
rows = []
for samples in dataset.values():
    for sample in samples:
        if hasattr(sample, "model_dump"):
            rows.append(sample.model_dump())
        elif hasattr(sample, "dict"):
            rows.append(sample.dict())
        elif isinstance(sample, dict):
            rows.append(sample)
        else:
            rows.append(vars(sample))
marker = {
    "version": 1,
    "dataset": "gpqa_diamond",
    "dataset_id": "AI-ModelScope/gpqa_diamond",
    "sample_count": sample_count,
}
Path(marker_path).parent.mkdir(parents=True, exist_ok=True)
Path(marker_path).write_text(json.dumps(marker, indent=2), encoding="utf-8")
Path(rows_path).write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print("gpqa_diamond_samples=" + str(sample_count))
"#;

    run_managed_child(
        &venv_python.to_string_lossy(),
        vec![
            "-c".to_string(),
            script.to_string(),
            dataset_root.to_string_lossy().to_string(),
            gpqa_dataset_marker_path(&app_data_dir)
                .to_string_lossy()
                .to_string(),
            gpqa_dataset_rows_path(&app_data_dir)
                .to_string_lossy()
                .to_string(),
        ],
        &child_slot,
    )?;

    progress.emit(
        crate::progress::ProgressStage::Benchmarking,
        1.0,
        "GPQA Diamond dataset downloaded and verified.",
    );

    Ok(detect_gpqa_diamond_status(app_data_dir))
}

fn download_humaneval_dataset_blocking(
    app_data_dir: PathBuf,
    app: tauri::AppHandle,
    child_slot: Arc<Mutex<Option<Child>>>,
) -> Result<HumanEvalDatasetStatus, String> {
    let venv_python = venv_python_path(&gpqa_env_dir(&app_data_dir));
    if !venv_python.exists() {
        return Err(
            "HumanEval harness is not installed. Install the harness before downloading the dataset."
                .to_string(),
        );
    }

    let progress = crate::progress::ProgressEmitter::new(app);
    let dataset_root = humaneval_dataset_cache_root(&app_data_dir);
    std::fs::create_dir_all(&dataset_root).map_err(|e| e.to_string())?;
    progress.emit(
        crate::progress::ProgressStage::Benchmarking,
        0.2,
        "Materializing HumanEval through EvalScope...",
    );

    let script = r#"
import json
import sys
from pathlib import Path

dataset_root, marker_path, rows_path = sys.argv[1], sys.argv[2], sys.argv[3]
from evalscope.config import TaskConfig
from evalscope.benchmarks.humaneval.humaneval_adapter import HumanevalAdapter
from evalscope.api.benchmark import BenchmarkMeta

meta = BenchmarkMeta(
    name="humaneval",
    dataset_id="opencompass/humaneval",
    data_adapter=HumanevalAdapter,
    subset_list=["openai_humaneval"],
    default_subset="openai_humaneval",
    eval_split="test",
    few_shot_num=0,
    prompt_template="Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.\n{question}",
)
task_config = TaskConfig(
    model="modelinspector-dataset-check",
    eval_type="mock_llm",
    datasets=["humaneval"],
    dataset_dir=dataset_root,
)
adapter = HumanevalAdapter(benchmark_meta=meta, task_config=task_config)
dataset = adapter.load_dataset()
sample_count = sum(len(samples) for samples in dataset.values())
if sample_count != 164:
    raise SystemExit(f"Expected 164 HumanEval samples, got {sample_count}")
rows = []
for samples in dataset.values():
    for sample in samples:
        if hasattr(sample, "model_dump"):
            rows.append(sample.model_dump())
        elif hasattr(sample, "dict"):
            rows.append(sample.dict())
        elif isinstance(sample, dict):
            rows.append(sample)
        else:
            rows.append(vars(sample))
marker = {
    "version": 1,
    "dataset": "humaneval",
    "dataset_id": "opencompass/humaneval",
    "sample_count": sample_count,
}
Path(marker_path).write_text(json.dumps(marker, indent=2), encoding="utf-8")
Path(rows_path).write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print("humaneval_samples=" + str(sample_count))
"#;

    run_managed_child(
        &venv_python.to_string_lossy(),
        vec![
            "-c".to_string(),
            script.to_string(),
            dataset_root.to_string_lossy().to_string(),
            humaneval_dataset_marker_path(&app_data_dir)
                .to_string_lossy()
                .to_string(),
            humaneval_dataset_rows_path(&app_data_dir)
                .to_string_lossy()
                .to_string(),
        ],
        &child_slot,
    )?;

    progress.emit(
        crate::progress::ProgressStage::Benchmarking,
        1.0,
        "HumanEval dataset downloaded and verified.",
    );
    Ok(detect_humaneval_dataset_status(&app_data_dir))
}

fn run_gpqa_diamond_blocking(
    base_url: String,
    api_key: String,
    model_id: String,
    shot_mode: GpqaShotMode,
    config: Option<GpqaRunConfig>,
    tensor_summary: ModelInspectorApiTensorSummary,
    model_summary: ModelInspectorApiModelSummary,
    runtime_totals: Arc<Mutex<ModelInspectorApiRuntimeTotals>>,
    app: tauri::AppHandle,
    child_slot: Arc<Mutex<Option<Child>>>,
) -> Result<BenchmarkResult, String> {
    let effective_config = effective_gpqa_run_config(config)?;
    let app_data_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to resolve app data directory: {e}"))?;
    let status = detect_gpqa_diamond_status(app_data_dir.clone());
    if !status.ready {
        return Err(format!(
            "GPQA Diamond is not ready. Current status: {}. {}",
            status.status_label, status.detail
        ));
    }

    let env_dir = gpqa_env_dir(&app_data_dir);
    let venv_python = venv_python_path(&env_dir);
    let evalscope_cli = evalscope_cli_path(&env_dir);
    if !venv_python.exists() {
        return Err(
            "EvalScope GPQA Diamond harness is not installed. Install it from Model Evaluation first."
                .to_string(),
        );
    }

    let run_dir = gpqa_run_dir(&app_data_dir).join(format!(
        "gpqa-diamond-{}-{}",
        shot_mode.few_shot_num(),
        unix_millis()
    ));
    std::fs::create_dir_all(&run_dir).map_err(|e| e.to_string())?;

    let dataset_args = json!({
        "gpqa_diamond": {
            "few_shot_num": shot_mode.few_shot_num()
        }
    })
    .to_string();
    let generation_config = gpqa_generation_config(&effective_config).to_string();

    let mut command = if evalscope_cli.exists() {
        Command::new(&evalscope_cli)
    } else {
        let mut fallback = Command::new(&venv_python);
        fallback.args(["-m", "evalscope"]);
        fallback
    };
    hide_child_console(&mut command);
    command
        .args([
            "eval".to_string(),
            "--eval-type".to_string(),
            "openai_api".to_string(),
            "--model".to_string(),
            model_id.clone(),
            "--model-id".to_string(),
            "modelinspector-gpqa-diamond".to_string(),
            "--api-url".to_string(),
            base_url.clone(),
            "--api-key".to_string(),
            api_key.clone(),
            "--datasets".to_string(),
            "gpqa_diamond".to_string(),
            "--dataset-dir".to_string(),
            gpqa_dataset_cache_root(&app_data_dir)
                .to_string_lossy()
                .to_string(),
            "--dataset-args".to_string(),
            dataset_args,
            "--generation-config".to_string(),
            generation_config,
            "--limit".to_string(),
            effective_config.sample_limit.to_string(),
            "--eval-batch-size".to_string(),
            "1".to_string(),
            "--repeats".to_string(),
            "1".to_string(),
            "--work-dir".to_string(),
            run_dir.to_string_lossy().to_string(),
            "--no-timestamp".to_string(),
            "--enable-progress-tracker".to_string(),
            "--no-collect-perf".to_string(),
        ])
        .env("PYTHONIOENCODING", "utf-8")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    crate::progress::emit_benchmark_output(
        &app,
        format!(
            "EvalScope: starting GPQA Diamond official harness ({})",
            shot_mode.label()
        ),
    );
    crate::progress::emit_benchmark_output(
        &app,
        format!("EvalScope: work directory {}", run_dir.display()),
    );
    crate::progress::ProgressEmitter::new(app.clone()).emit(
        crate::progress::ProgressStage::Benchmarking,
        0.05,
        "GPQA running",
    );

    let start = Instant::now();
    let mut child = command
        .spawn()
        .map_err(|e| format!("Failed to start GPQA Diamond harness: {e}"))?;
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    {
        let mut guard = child_slot.lock().map_err(|e| e.to_string())?;
        if guard.is_some() {
            let _ = child.kill();
            return Err("An official benchmark is already running.".to_string());
        }
        *guard = Some(child);
    }

    let stdout_handle = read_pipe_streaming(stdout, app.clone(), Some("EvalScope stdout"));
    let stderr_handle = read_pipe_streaming(stderr, app.clone(), Some("EvalScope stderr"));
    let status = loop {
        {
            let mut guard = child_slot.lock().map_err(|e| e.to_string())?;
            let child = guard
                .as_mut()
                .ok_or("Official benchmark process was not available.")?;
            if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
                break status;
            }
        }
        thread::sleep(Duration::from_millis(100));
    };

    {
        let mut guard = child_slot.lock().map_err(|e| e.to_string())?;
        let _ = guard.take();
    }

    let stdout = stdout_handle.join().unwrap_or_default();
    let stderr = stderr_handle.join().unwrap_or_default();
    let output = format!("{stdout}\n{stderr}");
    if !status.success() {
        if output.to_lowercase().contains("cancel") {
            crate::progress::emit_benchmark_output(
                &app,
                "EvalScope: GPQA Diamond official harness cancelled",
            );
            return Err("GPQA Diamond benchmark cancelled".to_string());
        }
        crate::progress::emit_benchmark_output(
            &app,
            format!("EvalScope: GPQA Diamond official harness failed with status {status}"),
        );
        return Err(format!(
            "GPQA Diamond harness failed with status {status}: {}",
            output.trim()
        ));
    }

    let report_path = find_gpqa_report_path(&run_dir).ok_or_else(|| {
        format!(
            "EvalScope finished but did not write a GPQA report under {}",
            run_dir.display()
        )
    })?;
    crate::progress::emit_benchmark_output(
        &app,
        format!("EvalScope: GPQA Diamond report {}", report_path.display()),
    );
    crate::progress::emit_benchmark_output(
        &app,
        "EvalScope: GPQA Diamond official harness finished",
    );
    Ok(gpqa_result_from_report(
        &model_id,
        shot_mode,
        &report_path,
        start.elapsed().as_millis() as f64,
        tensor_summary,
        model_summary,
        runtime_totals.lock().map_err(|e| e.to_string())?.snapshot(),
    )?)
}

fn run_humaneval_blocking(
    base_url: String,
    api_key: String,
    model_id: String,
    config: Option<GpqaRunConfig>,
    tensor_summary: ModelInspectorApiTensorSummary,
    model_summary: ModelInspectorApiModelSummary,
    runtime_totals: Arc<Mutex<ModelInspectorApiRuntimeTotals>>,
    app: tauri::AppHandle,
    child_slot: Arc<Mutex<Option<Child>>>,
) -> Result<BenchmarkResult, String> {
    let mut effective_config = effective_gpqa_run_config(config)?;
    if effective_config.sample_limit > HUMANEVAL_SAMPLE_COUNT {
        effective_config.sample_limit = HUMANEVAL_SAMPLE_COUNT;
    }
    let app_data_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to resolve app data directory: {e}"))?;
    let status = detect_humaneval_status(app_data_dir.clone());
    if !status.ready {
        return Err(format!(
            "HumanEval is not ready. Current status: {}. {}",
            status.status_label, status.detail
        ));
    }
    let dataset_status = detect_humaneval_dataset_status(&app_data_dir);
    if !dataset_status.dataset_ready {
        return Err("HumanEval dataset is not downloaded or verified yet.".to_string());
    }

    let env_dir = gpqa_env_dir(&app_data_dir);
    let venv_python = venv_python_path(&env_dir);
    let evalscope_cli = evalscope_cli_path(&env_dir);
    let run_dir = gpqa_run_dir(&app_data_dir).join(format!("humaneval-{}", unix_millis()));
    std::fs::create_dir_all(&run_dir).map_err(|e| e.to_string())?;

    let generation_config = gpqa_generation_config(&effective_config).to_string();
    let mut command = if evalscope_cli.exists() {
        Command::new(&evalscope_cli)
    } else {
        let mut fallback = Command::new(&venv_python);
        fallback.args(["-m", "evalscope"]);
        fallback
    };
    hide_child_console(&mut command);
    command
        .args([
            "eval".to_string(),
            "--eval-type".to_string(),
            "openai_api".to_string(),
            "--model".to_string(),
            model_id.clone(),
            "--model-id".to_string(),
            "modelinspector-humaneval".to_string(),
            "--api-url".to_string(),
            base_url.clone(),
            "--api-key".to_string(),
            api_key.clone(),
            "--datasets".to_string(),
            "humaneval".to_string(),
            "--dataset-dir".to_string(),
            humaneval_dataset_cache_root(&app_data_dir)
                .to_string_lossy()
                .to_string(),
            "--generation-config".to_string(),
            generation_config,
            "--limit".to_string(),
            effective_config.sample_limit.to_string(),
            "--eval-batch-size".to_string(),
            "1".to_string(),
            "--repeats".to_string(),
            "1".to_string(),
            "--work-dir".to_string(),
            run_dir.to_string_lossy().to_string(),
            "--no-timestamp".to_string(),
            "--enable-progress-tracker".to_string(),
            "--no-collect-perf".to_string(),
            "--sandbox".to_string(),
            r#"{"enabled": true, "engine": "docker"}"#.to_string(),
        ])
        .env("PYTHONIOENCODING", "utf-8")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    crate::progress::emit_benchmark_output(&app, "EvalScope: starting HumanEval official harness");
    crate::progress::emit_benchmark_output(
        &app,
        format!("EvalScope: work directory {}", run_dir.display()),
    );
    crate::progress::ProgressEmitter::new(app.clone()).emit(
        crate::progress::ProgressStage::Benchmarking,
        0.05,
        "HumanEval running",
    );

    let start = Instant::now();
    let mut child = command
        .spawn()
        .map_err(|e| format!("Failed to start HumanEval harness: {e}"))?;
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    {
        let mut guard = child_slot.lock().map_err(|e| e.to_string())?;
        if guard.is_some() {
            let _ = child.kill();
            return Err("An official benchmark is already running.".to_string());
        }
        *guard = Some(child);
    }

    let stdout_handle = read_pipe_streaming(stdout, app.clone(), Some("EvalScope stdout"));
    let stderr_handle = read_pipe_streaming(stderr, app.clone(), Some("EvalScope stderr"));
    let status = loop {
        {
            let mut guard = child_slot.lock().map_err(|e| e.to_string())?;
            let child = guard
                .as_mut()
                .ok_or("Official benchmark process was not available.")?;
            if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
                break status;
            }
        }
        thread::sleep(Duration::from_millis(100));
    };

    {
        let mut guard = child_slot.lock().map_err(|e| e.to_string())?;
        let _ = guard.take();
    }

    let stdout = stdout_handle.join().unwrap_or_default();
    let stderr = stderr_handle.join().unwrap_or_default();
    let output = format!("{stdout}\n{stderr}");
    if !status.success() {
        if output.to_lowercase().contains("cancel") {
            crate::progress::emit_benchmark_output(
                &app,
                "EvalScope: HumanEval official harness cancelled",
            );
            return Err("HumanEval benchmark cancelled".to_string());
        }
        crate::progress::emit_benchmark_output(
            &app,
            format!("EvalScope: HumanEval official harness failed with status {status}"),
        );
        return Err(format!(
            "HumanEval harness failed with status {status}: {}",
            output.trim()
        ));
    }

    let report_path = find_humaneval_report_path(&run_dir).ok_or_else(|| {
        format!(
            "EvalScope finished but did not write a HumanEval report under {}",
            run_dir.display()
        )
    })?;
    crate::progress::emit_benchmark_output(
        &app,
        format!("EvalScope: HumanEval report {}", report_path.display()),
    );
    crate::progress::emit_benchmark_output(&app, "EvalScope: HumanEval official harness finished");
    Ok(humaneval_result_from_report(
        &model_id,
        &report_path,
        start.elapsed().as_millis() as f64,
        tensor_summary,
        model_summary,
        runtime_totals.lock().map_err(|e| e.to_string())?.snapshot(),
    )?)
}

fn read_pipe_streaming(
    pipe: Option<impl Read + Send + 'static>,
    app: tauri::AppHandle,
    stream_label: Option<&'static str>,
) -> thread::JoinHandle<String> {
    thread::spawn(move || {
        let Some(pipe) = pipe else {
            return String::new();
        };
        let mut output = String::new();
        let reader = BufReader::new(pipe);
        for line in reader.lines() {
            let Ok(line) = line else {
                continue;
            };
            if output.is_empty() {
                output.push_str(&line);
            } else {
                output.push('\n');
                output.push_str(&line);
            }
            match stream_label {
                Some(label) => {
                    crate::progress::emit_benchmark_output(&app, format!("{label}: {line}"));
                }
                None => crate::progress::emit_benchmark_output(&app, line),
            }
        }
        output
    })
}

fn read_pipe_async(pipe: Option<impl Read + Send + 'static>) -> thread::JoinHandle<String> {
    thread::spawn(move || {
        let Some(mut pipe) = pipe else {
            return String::new();
        };
        let mut output = String::new();
        let _ = pipe.read_to_string(&mut output);
        output
    })
}

fn find_python() -> Option<PythonCommand> {
    let candidates = [
        PythonCommand {
            executable: "python".to_string(),
            prefix_args: Vec::new(),
        },
        PythonCommand {
            executable: "python3".to_string(),
            prefix_args: Vec::new(),
        },
        PythonCommand {
            executable: "py".to_string(),
            prefix_args: vec!["-3".to_string()],
        },
    ];

    candidates.into_iter().find(|candidate| {
        let mut command = Command::new(&candidate.executable);
        hide_child_console(&mut command);
        command.args(&candidate.prefix_args).arg("--version");
        command
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    })
}

fn gpqa_env_dir(app_data_dir: &Path) -> PathBuf {
    if let Some(local_app_data) = std::env::var_os("LOCALAPPDATA") {
        return PathBuf::from(local_app_data).join("MI").join("g");
    }

    app_data_dir.join("b").join("g")
}

fn gpqa_dataset_cache_root(app_data_dir: &Path) -> PathBuf {
    gpqa_env_dir(app_data_dir).join("datasets")
}

fn gpqa_dataset_marker_path(app_data_dir: &Path) -> PathBuf {
    gpqa_dataset_cache_root(app_data_dir).join("gpqa_diamond_dataset_ready.json")
}

fn gpqa_dataset_rows_path(app_data_dir: &Path) -> PathBuf {
    gpqa_dataset_cache_root(app_data_dir).join("gpqa_diamond_rows.json")
}

fn humaneval_dataset_cache_root(app_data_dir: &Path) -> PathBuf {
    gpqa_dataset_cache_root(app_data_dir).join("humaneval")
}

fn humaneval_dataset_marker_path(app_data_dir: &Path) -> PathBuf {
    humaneval_dataset_cache_root(app_data_dir).join("humaneval_dataset_ready.json")
}

fn humaneval_dataset_rows_path(app_data_dir: &Path) -> PathBuf {
    humaneval_dataset_cache_root(app_data_dir).join("humaneval_rows.json")
}

fn gpqa_run_dir(app_data_dir: &Path) -> PathBuf {
    gpqa_env_dir(app_data_dir).join("runs")
}

fn venv_python_path(env_dir: &Path) -> PathBuf {
    if cfg!(windows) {
        env_dir.join("venv").join("Scripts").join("python.exe")
    } else {
        env_dir.join("venv").join("bin").join("python")
    }
}

fn evalscope_cli_path(env_dir: &Path) -> PathBuf {
    if cfg!(windows) {
        env_dir.join("venv").join("Scripts").join("evalscope.exe")
    } else {
        env_dir.join("venv").join("bin").join("evalscope")
    }
}

fn managed_env_status(
    venv_exists: bool,
    system_python_exists: bool,
    probe_output: Option<&str>,
    dataset_state: GpqaDatasetState,
) -> GpqaDiamondStatus {
    if let Some(output) = probe_output {
        return classify_gpqa_status(output, dataset_state);
    }

    if !venv_exists && system_python_exists {
        return GpqaDiamondStatus {
            ready: false,
            status_label: "Install".to_string(),
            python: None,
            evalscope: None,
            dataset_ready: matches!(dataset_state, GpqaDatasetState::Verified { .. }),
            dataset_status_label: dataset_status_label(&dataset_state).to_string(),
            dataset_path: dataset_path_string(&dataset_state),
            dataset_hash: dataset_hash_string(&dataset_state),
            dataset_url: GPQA_DATASET_ID.to_string(),
            expected_dataset_hash: "EvalScope dataset cache marker".to_string(),
            detail: "System Python is available, but the managed benchmark environment has not been created yet.".to_string(),
        };
    }

    GpqaDiamondStatus {
        ready: false,
        status_label: "Needs Python".to_string(),
        python: None,
        evalscope: None,
        dataset_ready: matches!(dataset_state, GpqaDatasetState::Verified { .. }),
        dataset_status_label: dataset_status_label(&dataset_state).to_string(),
        dataset_path: dataset_path_string(&dataset_state),
        dataset_hash: dataset_hash_string(&dataset_state),
        dataset_url: GPQA_DATASET_ID.to_string(),
        expected_dataset_hash: "EvalScope dataset cache marker".to_string(),
        detail: "Python was not found on PATH.".to_string(),
    }
}

fn run_managed_child(
    executable: &str,
    args: Vec<String>,
    child_slot: &Arc<Mutex<Option<Child>>>,
) -> Result<String, String> {
    let mut command = Command::new(executable);
    hide_child_console(&mut command);
    let mut child = command
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to start benchmark setup command: {e}"))?;
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    {
        let mut guard = child_slot.lock().map_err(|e| e.to_string())?;
        if guard.is_some() {
            let _ = child.kill();
            return Err("A benchmark setup or run is already active.".to_string());
        }
        *guard = Some(child);
    }

    let stdout_handle = read_pipe_async(stdout);
    let stderr_handle = read_pipe_async(stderr);
    let status = loop {
        {
            let mut guard = child_slot.lock().map_err(|e| e.to_string())?;
            let child = guard
                .as_mut()
                .ok_or("Benchmark setup process was not available.")?;
            if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
                break status;
            }
        }
        thread::sleep(Duration::from_millis(100));
    };

    {
        let mut guard = child_slot.lock().map_err(|e| e.to_string())?;
        let _ = guard.take();
    }

    let stdout = stdout_handle.join().unwrap_or_default();
    let stderr = stderr_handle.join().unwrap_or_default();
    let output = format!("{stdout}\n{stderr}");
    if status.success() {
        Ok(output)
    } else if output.to_lowercase().contains("cancel") {
        Err("Benchmark setup cancelled".to_string())
    } else {
        Err(format!("Benchmark setup command failed: {}", output.trim()))
    }
}

fn run_python_probe(python: &PythonCommand) -> Result<String, String> {
    let script = r#"
import sys
try:
    from importlib import metadata
except Exception:
    import importlib_metadata as metadata

print("python=" + ".".join(str(part) for part in sys.version_info[:3]))
try:
    import evalscope
    print("evalscope=" + metadata.version("evalscope"))
except Exception as exc:
    print("evalscope_error=" + str(exc))
try:
    from evalscope.benchmarks.gpqa.gpqa_adapter import GPQAAdapter
    print("gpqa_task=ok")
except Exception as exc:
    print("gpqa_task_error=" + str(exc))
try:
    import openai
    print("openai=" + metadata.version("openai"))
except Exception as exc:
    print("openai_error=" + str(exc))
"#;

    let mut command = Command::new(&python.executable);
    hide_child_console(&mut command);
    command.args(&python.prefix_args).args(["-c", script]);
    let output = command
        .output()
        .map_err(|e| format!("Failed to probe Python benchmark harness: {e}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if output.status.success() {
        Ok(stdout.to_string())
    } else {
        Err(format!(
            "Python benchmark harness probe failed: {}",
            stderr.trim()
        ))
    }
}

fn run_humaneval_probe(python: &PythonCommand) -> Result<String, String> {
    let script = r#"
import sys
try:
    from importlib import metadata
except Exception:
    import importlib_metadata as metadata

print("python=" + ".".join(str(part) for part in sys.version_info[:3]))
try:
    import evalscope
    print("evalscope=" + metadata.version("evalscope"))
except Exception as exc:
    print("evalscope_error=" + str(exc))
try:
    from evalscope.benchmarks.humaneval.humaneval_adapter import HumanevalAdapter
    print("humaneval_task=ok")
except Exception as exc:
    print("humaneval_task_error=" + str(exc))
try:
    import ms_enclave
    print("sandbox=ok")
except Exception as exc:
    print("sandbox_error=" + str(exc))
"#;

    let mut command = Command::new(&python.executable);
    hide_child_console(&mut command);
    command.args(&python.prefix_args).args(["-c", script]);
    let output = command
        .output()
        .map_err(|e| format!("Failed to probe HumanEval benchmark harness: {e}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if output.status.success() {
        Ok(stdout.to_string())
    } else {
        Err(format!(
            "HumanEval benchmark harness probe failed: {}",
            stderr.trim()
        ))
    }
}

fn detect_docker_status() -> (bool, Option<String>, String) {
    let mut command = Command::new("docker");
    hide_child_console(&mut command);
    let output = command
        .args(["info", "--format", "{{.ServerVersion}}"])
        .output();
    match output {
        Ok(output) if output.status.success() => {
            let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let version = if version.is_empty() {
                None
            } else {
                Some(version)
            };
            (
                true,
                version.clone(),
                version
                    .map(|value| format!("Docker daemon is available ({value})."))
                    .unwrap_or_else(|| "Docker daemon is available.".to_string()),
            )
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            (
                false,
                None,
                if stderr.is_empty() {
                    "Docker CLI is available, but the Docker daemon is not running.".to_string()
                } else {
                    stderr
                },
            )
        }
        Err(error) => (
            false,
            None,
            format!("Docker was not found or could not be started: {error}"),
        ),
    }
}

#[cfg(windows)]
fn hide_child_console(command: &mut Command) {
    use std::os::windows::process::CommandExt;
    command.creation_flags(CREATE_NO_WINDOW);
}

#[cfg(not(windows))]
fn hide_child_console(_: &mut Command) {}

#[cfg(test)]
fn classify_probe_output(output: &str) -> GpqaDiamondStatus {
    classify_gpqa_status(
        output,
        GpqaDatasetState::Verified {
            path: PathBuf::from("gpqa_diamond_dataset_ready.json"),
            hash: "marker-v1".to_string(),
        },
    )
}

fn classify_gpqa_status(output: &str, dataset_state: GpqaDatasetState) -> GpqaDiamondStatus {
    let python = probe_value(output, "python");
    let evalscope = probe_value(output, "evalscope");
    let gpqa_task = probe_value(output, "gpqa_task");
    let openai = probe_value(output, "openai");
    let harness_ready = python.is_some()
        && evalscope.is_some()
        && gpqa_task.as_deref() == Some("ok")
        && openai.is_some();
    let dataset_ready = matches!(dataset_state, GpqaDatasetState::Verified { .. });
    let ready = harness_ready && dataset_ready;

    GpqaDiamondStatus {
        ready,
        status_label: if !harness_ready {
            "Needs harness"
        } else if !dataset_ready {
            "Download"
        } else {
            "Ready"
        }
        .to_string(),
        python,
        evalscope,
        dataset_ready,
        dataset_status_label: dataset_status_label(&dataset_state).to_string(),
        dataset_path: dataset_path_string(&dataset_state),
        dataset_hash: dataset_hash_string(&dataset_state),
        dataset_url: GPQA_DATASET_ID.to_string(),
        expected_dataset_hash: "EvalScope dataset cache marker".to_string(),
        detail: if ready {
            "EvalScope, GPQA Diamond adapter, OpenAI client, and dataset cache marker are verified."
                .to_string()
        } else if harness_ready {
            dataset_detail(&dataset_state)
        } else {
            output
                .lines()
                .filter(|line| line.contains("_error="))
                .collect::<Vec<_>>()
                .join("; ")
        },
    }
}

fn classify_humaneval_status(
    output: &str,
    docker_ready: bool,
    docker: Option<String>,
    docker_detail: String,
) -> HumanEvalStatus {
    let python = probe_value(output, "python");
    let evalscope = probe_value(output, "evalscope");
    let humaneval_task = probe_value(output, "humaneval_task");
    let sandbox = probe_value(output, "sandbox");
    let harness_ready = python.is_some()
        && evalscope.is_some()
        && humaneval_task.as_deref() == Some("ok")
        && sandbox.as_deref() == Some("ok");
    let ready = harness_ready && docker_ready;

    HumanEvalStatus {
        ready,
        status_label: if !harness_ready {
            "Needs harness"
        } else if !docker_ready {
            "Needs Docker"
        } else {
            "Ready"
        }
        .to_string(),
        python,
        evalscope,
        docker_ready,
        docker,
        detail: if !harness_ready {
            output
                .lines()
                .filter(|line| line.contains("_error="))
                .collect::<Vec<_>>()
                .join("; ")
        } else if ready {
            "EvalScope HumanEval adapter and Docker daemon are verified.".to_string()
        } else if !docker_ready {
            docker_detail
        } else {
            String::new()
        },
    }
}

fn detect_humaneval_dataset_status(app_data_dir: &Path) -> HumanEvalDatasetStatus {
    let state = detect_humaneval_dataset_state(app_data_dir);
    HumanEvalDatasetStatus {
        dataset_ready: matches!(state, GpqaDatasetState::Verified { .. }),
        dataset_status_label: dataset_status_label(&state).to_string(),
        dataset_path: dataset_path_string(&state),
        dataset_hash: dataset_hash_string(&state),
        dataset_url: HUMANEVAL_DATASET_ID.to_string(),
        expected_dataset_hash: "EvalScope dataset cache marker".to_string(),
    }
}

fn detect_humaneval_dataset_state(app_data_dir: &Path) -> GpqaDatasetState {
    let path = humaneval_dataset_marker_path(app_data_dir);
    if !path.exists() {
        return GpqaDatasetState::Missing;
    }

    match std::fs::read_to_string(&path)
        .map_err(|e| e.to_string())
        .and_then(|text| {
            serde_json::from_str::<serde_json::Value>(&text).map_err(|e| e.to_string())
        }) {
        Ok(marker)
            if marker.get("version").and_then(|value| value.as_u64())
                == Some(HUMANEVAL_DATASET_MARKER_VERSION as u64)
                && marker.get("dataset").and_then(|value| value.as_str()) == Some("humaneval")
                && marker.get("sample_count").and_then(|value| value.as_u64())
                    == Some(HUMANEVAL_SAMPLE_COUNT) =>
        {
            GpqaDatasetState::Verified {
                path,
                hash: format!("marker-v{HUMANEVAL_DATASET_MARKER_VERSION}"),
            }
        }
        Ok(_) => GpqaDatasetState::Invalid {
            path,
            hash: None,
            detail: "HumanEval EvalScope dataset marker is invalid.".to_string(),
        },
        Err(error) => GpqaDatasetState::Invalid {
            path,
            hash: None,
            detail: error,
        },
    }
}

fn detect_gpqa_dataset_state(app_data_dir: &Path) -> GpqaDatasetState {
    let path = gpqa_dataset_marker_path(app_data_dir);
    if !path.exists() {
        return GpqaDatasetState::Missing;
    }

    match std::fs::read_to_string(&path)
        .map_err(|e| e.to_string())
        .and_then(|text| {
            serde_json::from_str::<serde_json::Value>(&text).map_err(|e| e.to_string())
        }) {
        Ok(marker)
            if marker.get("version").and_then(|value| value.as_u64())
                == Some(GPQA_DATASET_MARKER_VERSION as u64)
                && marker.get("dataset").and_then(|value| value.as_str())
                    == Some("gpqa_diamond")
                && marker.get("sample_count").and_then(|value| value.as_u64())
                    == Some(GPQA_SAMPLE_COUNT) =>
        {
            GpqaDatasetState::Verified {
                path,
                hash: format!("marker-v{GPQA_DATASET_MARKER_VERSION}"),
            }
        }
        Ok(_) => GpqaDatasetState::Invalid {
            path,
            hash: None,
            detail: "GPQA Diamond EvalScope dataset marker is invalid.".to_string(),
        },
        Err(error) => GpqaDatasetState::Invalid {
            path,
            hash: None,
            detail: error,
        },
    }
}

fn dataset_status_label(state: &GpqaDatasetState) -> &'static str {
    match state {
        GpqaDatasetState::Missing => "Missing",
        GpqaDatasetState::Verified { .. } => "Verified",
        GpqaDatasetState::Invalid { .. } => "Invalid",
    }
}

fn dataset_path_string(state: &GpqaDatasetState) -> Option<String> {
    match state {
        GpqaDatasetState::Missing => None,
        GpqaDatasetState::Verified { path, .. } | GpqaDatasetState::Invalid { path, .. } => {
            Some(path.to_string_lossy().to_string())
        }
    }
}

fn dataset_hash_string(state: &GpqaDatasetState) -> Option<String> {
    match state {
        GpqaDatasetState::Missing => None,
        GpqaDatasetState::Verified { hash, .. } => Some(hash.clone()),
        GpqaDatasetState::Invalid { hash, .. } => hash.clone(),
    }
}

fn dataset_detail(state: &GpqaDatasetState) -> String {
    match state {
        GpqaDatasetState::Missing => {
            "GPQA Diamond dataset is not downloaded or verified yet.".to_string()
        }
        GpqaDatasetState::Verified { .. } => {
            "GPQA Diamond dataset is downloaded and hash verified.".to_string()
        }
        GpqaDatasetState::Invalid { detail, .. } => detail.clone(),
    }
}

fn read_gpqa_dataset_rows(app_data_dir: &Path) -> Result<Vec<GpqaDatasetRow>, String> {
    if !matches!(
        detect_gpqa_dataset_state(app_data_dir),
        GpqaDatasetState::Verified { .. }
    ) {
        return Err("GPQA Diamond dataset is not downloaded or verified yet.".to_string());
    }

    let text = std::fs::read_to_string(gpqa_dataset_rows_path(app_data_dir)).map_err(|e| {
        format!("Failed to read GPQA Diamond dataset rows: {e}. Click Download dataset to refresh the preview.")
    })?;
    let rows_json: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("Failed to parse GPQA rows JSON: {e}"))?;
    let rows = rows_json
        .as_array()
        .ok_or_else(|| "GPQA rows JSON must be an array.".to_string())?
        .iter()
        .enumerate()
        .map(|(index, row)| gpqa_dataset_row_from_json(index + 1, row))
        .collect();
    Ok(rows)
}

fn gpqa_dataset_row_from_json(index: usize, row: &serde_json::Value) -> GpqaDatasetRow {
    GpqaDatasetRow {
        index,
        question: string_field(row, &["question", "Question", "problem", "query", "prompt"])
            .or_else(|| gpqa_question_from_input(row))
            .unwrap_or_default(),
        choices: choices_field(row, &["choices", "Choices", "options", "Options"]),
        answer: row
            .get("metadata")
            .and_then(|metadata| string_field(metadata, &["correct_answer", "correctAnswer"]))
            .or_else(|| {
                string_field(
                    row,
                    &[
                        "answer",
                        "Answer",
                        "target",
                        "gold",
                        "label",
                        "correct_answer",
                    ],
                )
            }),
    }
}

fn read_humaneval_dataset_rows(app_data_dir: &Path) -> Result<Vec<HumanEvalDatasetRow>, String> {
    if !matches!(
        detect_humaneval_dataset_state(app_data_dir),
        GpqaDatasetState::Verified { .. }
    ) {
        return Err("HumanEval dataset is not downloaded or verified yet.".to_string());
    }

    let text = std::fs::read_to_string(humaneval_dataset_rows_path(app_data_dir)).map_err(|e| {
        format!(
            "Failed to read HumanEval dataset rows: {e}. Click Download dataset to refresh the preview."
        )
    })?;
    let rows_json: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| format!("Failed to parse HumanEval rows JSON: {e}"))?;
    rows_json
        .as_array()
        .ok_or_else(|| "HumanEval rows JSON must be an array.".to_string())
        .map(|rows| {
            rows.iter()
                .enumerate()
                .map(|(index, row)| humaneval_dataset_row_from_json(index + 1, row))
                .collect()
        })
}

fn humaneval_dataset_row_from_json(index: usize, row: &serde_json::Value) -> HumanEvalDatasetRow {
    let metadata = row.get("metadata").unwrap_or(&serde_json::Value::Null);
    HumanEvalDatasetRow {
        index,
        task_id: string_field(metadata, &["task_id"]).unwrap_or_default(),
        entry_point: string_field(metadata, &["entry_point"]).unwrap_or_default(),
        prompt: string_field(metadata, &["prompt"]).unwrap_or_default(),
        canonical_solution: string_field(row, &["target", "canonical_solution"])
            .unwrap_or_default(),
    }
}

fn gpqa_question_from_input(row: &serde_json::Value) -> Option<String> {
    let content = row
        .get("input")?
        .as_array()?
        .first()?
        .get("content")?
        .as_str()?;
    let question = content
        .rsplit_once("Think step by step before answering.")?
        .1
        .split("\nA)")
        .next()?
        .trim();
    (!question.is_empty()).then(|| question.to_string())
}

fn string_field(row: &serde_json::Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .filter_map(|key| row.get(*key))
        .find_map(|value| value.as_str().map(str::to_string))
}

fn choices_field(row: &serde_json::Value, keys: &[&str]) -> Vec<String> {
    for key in keys {
        let Some(value) = row.get(*key) else {
            continue;
        };
        if let Some(values) = value.as_array() {
            return values
                .iter()
                .filter_map(|choice| choice.as_str().map(str::to_string))
                .collect();
        }
        if let Some(values) = value.as_object() {
            let mut choices: Vec<_> = values
                .iter()
                .filter_map(|(label, choice)| {
                    choice.as_str().map(|text| format!("{}. {}", label, text))
                })
                .collect();
            choices.sort();
            return choices;
        }
    }
    Vec::new()
}

fn probe_value(output: &str, key: &str) -> Option<String> {
    let prefix = format!("{key}=");
    output.lines().find_map(|line| {
        line.strip_prefix(&prefix)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
    })
}

fn find_gpqa_report_path(run_dir: &Path) -> Option<PathBuf> {
    fn visit(path: &Path) -> Option<PathBuf> {
        let entries = std::fs::read_dir(path).ok()?;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if let Some(found) = visit(&path) {
                    return Some(found);
                }
            } else if path.file_name().and_then(|name| name.to_str()) == Some("gpqa_diamond.json") {
                return Some(path);
            }
        }
        None
    }

    visit(&run_dir.join("reports")).or_else(|| visit(run_dir))
}

fn find_humaneval_report_path(run_dir: &Path) -> Option<PathBuf> {
    fn visit(dir: &Path) -> Option<PathBuf> {
        let entries = std::fs::read_dir(dir).ok()?;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if let Some(found) = visit(&path) {
                    return Some(found);
                }
            } else if path.file_name().and_then(|name| name.to_str()) == Some("humaneval.json") {
                return Some(path);
            }
        }
        None
    }

    visit(&run_dir.join("reports")).or_else(|| visit(run_dir))
}

fn gpqa_result_from_report(
    model_id: &str,
    shot_mode: GpqaShotMode,
    report_path: &Path,
    elapsed_ms: f64,
    tensor_summary: ModelInspectorApiTensorSummary,
    model_summary: ModelInspectorApiModelSummary,
    runtime_summary: ModelInspectorApiRuntimeSummary,
) -> Result<BenchmarkResult, String> {
    let report_text = std::fs::read_to_string(report_path)
        .map_err(|e| format!("Failed to read EvalScope GPQA report: {e}"))?;
    let report_json: serde_json::Value = serde_json::from_str(&report_text)
        .map_err(|e| format!("Failed to parse EvalScope GPQA report JSON: {e}"))?;
    let accuracy = extract_accuracy_from_json(&report_json).unwrap_or(0.0);
    let metric = extract_metric_from_json(&report_json).unwrap_or_else(|| "mean_acc".to_string());
    let sample_count = extract_sample_count_from_json(&report_json).unwrap_or(GPQA_SAMPLE_COUNT);
    Ok(BenchmarkResult {
        prompt_eval_tps: runtime_summary.prompt_eval_tps,
        token_gen_tps: runtime_summary.token_gen_tps,
        ttft_ms: runtime_summary.ttft_ms,
        prompt_eval_ms: runtime_summary.prompt_eval_ms,
        generation_ms: runtime_summary.generation_ms,
        vram_peak_mb: runtime_summary.vram_peak_mb,
        vram_allocated_mb: runtime_summary.vram_allocated_mb,
        disk_size_mb: 0.0,
        elapsed_ms,
        load_ms: runtime_summary.load_ms,
        test_mode: "official_gpqa_diamond".to_string(),
        status_message: format!(
            "GPQA Diamond EvalScope official harness completed for {model_id} with {}.",
            shot_mode.label()
        ),
        native_runtime: Some(format!(
            "EvalScope GPQA Diamond report: {}",
            report_path.display()
        )),
        model_tensor_count: Some(model_summary.tensor_count),
        model_metadata_count: Some(model_summary.metadata_count),
        copied_tensor_count: tensor_summary.copied_tensor_count,
        converted_tensor_count: tensor_summary.converted_tensor_count,
        converted_bytes_before: tensor_summary.converted_bytes_before,
        converted_bytes_after: tensor_summary.converted_bytes_after,
        requested_target_count: tensor_summary.requested_target_count,
        verified_target_count: tensor_summary.verified_target_count,
        baseline_benchmark: None,
        quality_eval: None,
        standard_eval: Some(StandardEvalReport {
            sample_count,
            task_count: 1,
            baseline_accuracy: None,
            recipe_accuracy: accuracy,
            accuracy_delta: None,
            correct_to_wrong_count: 0,
            wrong_to_correct_count: 0,
            baseline_avg_margin: None,
            recipe_avg_margin: 0.0,
            margin_delta: None,
            tasks: vec![StandardEvalTaskReport {
                task: "gpqa_diamond".to_string(),
                metric,
                n_shot: shot_mode.few_shot_num() as u64,
                sample_count,
                baseline_correct_count: None,
                recipe_correct_count: (accuracy * sample_count as f64).round() as u64,
                correct_to_wrong_count: 0,
                wrong_to_correct_count: 0,
                same_prediction_count: 0,
                baseline_accuracy: None,
                recipe_accuracy: accuracy,
                accuracy_delta: None,
                baseline_avg_margin: None,
                recipe_avg_margin: 0.0,
                margin_delta: None,
                baseline_avg_correct_nll: None,
                recipe_avg_correct_nll: 0.0,
            }],
            sample_audits: Vec::new(),
        }),
    })
}

fn humaneval_result_from_report(
    model_id: &str,
    report_path: &Path,
    elapsed_ms: f64,
    tensor_summary: ModelInspectorApiTensorSummary,
    model_summary: ModelInspectorApiModelSummary,
    runtime_summary: ModelInspectorApiRuntimeSummary,
) -> Result<BenchmarkResult, String> {
    let report_text = std::fs::read_to_string(report_path)
        .map_err(|e| format!("Failed to read EvalScope HumanEval report: {e}"))?;
    let report_json: serde_json::Value = serde_json::from_str(&report_text)
        .map_err(|e| format!("Failed to parse EvalScope HumanEval report JSON: {e}"))?;
    let pass_at_1 = extract_humaneval_pass_at_1(&report_json).unwrap_or(0.0);
    let sample_count = extract_sample_count_from_json(&report_json).unwrap_or(HUMANEVAL_SAMPLE_COUNT);
    Ok(BenchmarkResult {
        prompt_eval_tps: runtime_summary.prompt_eval_tps,
        token_gen_tps: runtime_summary.token_gen_tps,
        ttft_ms: runtime_summary.ttft_ms,
        prompt_eval_ms: runtime_summary.prompt_eval_ms,
        generation_ms: runtime_summary.generation_ms,
        vram_peak_mb: runtime_summary.vram_peak_mb,
        vram_allocated_mb: runtime_summary.vram_allocated_mb,
        disk_size_mb: 0.0,
        elapsed_ms,
        load_ms: runtime_summary.load_ms,
        test_mode: "official_humaneval".to_string(),
        status_message: format!("HumanEval EvalScope official harness completed for {model_id}."),
        native_runtime: Some(format!("EvalScope HumanEval report: {}", report_path.display())),
        model_tensor_count: Some(model_summary.tensor_count),
        model_metadata_count: Some(model_summary.metadata_count),
        copied_tensor_count: tensor_summary.copied_tensor_count,
        converted_tensor_count: tensor_summary.converted_tensor_count,
        converted_bytes_before: tensor_summary.converted_bytes_before,
        converted_bytes_after: tensor_summary.converted_bytes_after,
        requested_target_count: tensor_summary.requested_target_count,
        verified_target_count: tensor_summary.verified_target_count,
        baseline_benchmark: None,
        quality_eval: None,
        standard_eval: Some(StandardEvalReport {
            sample_count,
            task_count: 1,
            baseline_accuracy: None,
            recipe_accuracy: pass_at_1,
            accuracy_delta: None,
            correct_to_wrong_count: 0,
            wrong_to_correct_count: 0,
            baseline_avg_margin: None,
            recipe_avg_margin: 0.0,
            margin_delta: None,
            tasks: vec![StandardEvalTaskReport {
                task: "humaneval".to_string(),
                metric: "pass@1".to_string(),
                n_shot: 0,
                sample_count,
                baseline_correct_count: None,
                recipe_correct_count: (pass_at_1 * sample_count as f64).round() as u64,
                correct_to_wrong_count: 0,
                wrong_to_correct_count: 0,
                same_prediction_count: 0,
                baseline_accuracy: None,
                recipe_accuracy: pass_at_1,
                accuracy_delta: None,
                baseline_avg_margin: None,
                recipe_avg_margin: 0.0,
                margin_delta: None,
                baseline_avg_correct_nll: None,
                recipe_avg_correct_nll: 0.0,
            }],
            sample_audits: Vec::new(),
        }),
    })
}

fn extract_humaneval_pass_at_1(value: &serde_json::Value) -> Option<f64> {
    match value {
        serde_json::Value::Object(map) => {
            for (key, candidate) in map {
                let key = key.to_lowercase();
                if matches!(key.as_str(), "pass@1" | "pass_at_1" | "pass@k" | "score") {
                    if let Some(score) = normalize_score(candidate.as_f64()) {
                        return Some(score);
                    }
                }
            }
            map.values().find_map(extract_humaneval_pass_at_1)
        }
        serde_json::Value::Array(items) => items.iter().find_map(extract_humaneval_pass_at_1),
        _ => None,
    }
}

fn extract_accuracy_from_json(value: &serde_json::Value) -> Option<f64> {
    match value {
        serde_json::Value::Object(map) => {
            for (key, candidate) in map {
                let key = key.to_lowercase();
                if matches!(
                    key.as_str(),
                    "mean_acc" | "acc_norm" | "acc" | "accuracy" | "score" | "value"
                ) {
                    if let Some(score) = normalize_score(candidate.as_f64()) {
                        return Some(score);
                    }
                }
            }
            map.values().find_map(extract_accuracy_from_json)
        }
        serde_json::Value::Array(items) => items.iter().find_map(extract_accuracy_from_json),
        _ => None,
    }
}

fn extract_metric_from_json(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::Object(map) => {
            if let Some(metric) = map
                .get("metric")
                .and_then(|value| value.as_str())
                .filter(|value| !value.trim().is_empty())
            {
                return Some(metric.to_string());
            }
            if let Some(metrics) = map.get("metrics").and_then(|value| value.as_object()) {
                for metric in ["mean_acc", "acc_norm", "acc", "accuracy", "score", "value"] {
                    if metrics
                        .get(metric)
                        .and_then(|value| value.as_f64())
                        .is_some()
                    {
                        return Some(metric.to_string());
                    }
                }
            }
            map.values().find_map(extract_metric_from_json)
        }
        serde_json::Value::Array(items) => items.iter().find_map(extract_metric_from_json),
        _ => None,
    }
}

fn extract_sample_count_from_json(value: &serde_json::Value) -> Option<u64> {
    match value {
        serde_json::Value::Object(map) => {
            for (key, candidate) in map {
                let key = key.to_lowercase();
                if matches!(
                    key.as_str(),
                    "sample_count" | "num_samples" | "total" | "num"
                ) && candidate.as_u64().is_some()
                {
                    return candidate.as_u64();
                }
            }
            map.values().find_map(extract_sample_count_from_json)
        }
        serde_json::Value::Array(items) => items.iter().find_map(extract_sample_count_from_json),
        _ => None,
    }
}

fn normalize_score(score: Option<f64>) -> Option<f64> {
    let score = score?;
    if score > 1.0 && score <= 100.0 {
        Some(score / 100.0)
    } else if (0.0..=1.0).contains(&score) {
        Some(score)
    } else {
        None
    }
}

fn unix_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_gpqa_ready_when_required_packages_are_importable() {
        let status =
            classify_probe_output("python=3.11.8\nevalscope=1.8.0\ngpqa_task=ok\nopenai=1.0.0\n");

        assert!(status.ready);
        assert_eq!(status.status_label, "Ready");
        assert_eq!(status.python.as_deref(), Some("3.11.8"));
        assert_eq!(status.evalscope.as_deref(), Some("1.8.0"));
    }

    #[test]
    fn classifies_gpqa_as_needing_harness_when_evalscope_is_missing() {
        let status = classify_probe_output(
            "python=3.11.8\nevalscope_error=No module named evalscope\nopenai=1.0.0\n",
        );

        assert!(!status.ready);
        assert_eq!(status.status_label, "Needs harness");
        assert!(status.detail.contains("evalscope_error"));
    }

    #[test]
    fn classifies_gpqa_as_needing_harness_when_task_is_missing() {
        let status = classify_probe_output(
            "python=3.11.8\nevalscope=1.8.0\nopenai=1.0.0\ngpqa_task_error=No module named evalscope.benchmarks.gpqa\n",
        );

        assert!(!status.ready);
        assert_eq!(status.status_label, "Needs harness");
        assert!(status.detail.contains("gpqa_task_error"));
    }

    #[test]
    fn classifies_humaneval_as_needing_harness_when_sandbox_is_missing() {
        let status = classify_humaneval_status(
            "python=3.11.8\nevalscope=1.8.0\nhumaneval_task=ok\nsandbox_error=No module named ms_enclave\n",
            true,
            Some("Docker".to_string()),
            String::new(),
        );

        assert!(!status.ready);
        assert_eq!(status.status_label, "Needs harness");
        assert!(status.detail.contains("sandbox_error"));
    }

    #[test]
    fn prioritizes_missing_humaneval_harness_over_missing_docker() {
        let status = classify_humaneval_status(
            "python=3.11.8\nevalscope=1.8.0\nhumaneval_task=ok\nsandbox_error=No module named ms_enclave\n",
            false,
            None,
            "Docker is not running.".to_string(),
        );

        assert!(!status.ready);
        assert_eq!(status.status_label, "Needs harness");
        assert!(status.detail.contains("sandbox_error"));
    }

    #[test]
    fn reports_install_when_system_python_exists_but_managed_env_is_missing() {
        let status = managed_env_status(false, true, None, GpqaDatasetState::Missing);

        assert!(!status.ready);
        assert_eq!(status.status_label, "Install");
        assert!(status.detail.contains("managed benchmark environment"));
    }

    #[test]
    fn reports_ready_from_managed_env_probe() {
        let status = managed_env_status(
            true,
            true,
            Some("python=3.11.8\nevalscope=1.8.0\ngpqa_task=ok\nopenai=1.0.0\n"),
            GpqaDatasetState::Verified {
                path: PathBuf::from("gpqa_diamond_dataset_ready.json"),
                hash: "marker-v1".to_string(),
            },
        );

        assert!(status.ready);
        assert_eq!(status.status_label, "Ready");
        assert_eq!(status.python.as_deref(), Some("3.11.8"));
    }

    #[test]
    fn reports_download_when_harness_is_ready_but_dataset_is_missing() {
        let status = classify_gpqa_status(
            "python=3.11.8\nevalscope=1.8.0\ngpqa_task=ok\nopenai=1.0.0\n",
            GpqaDatasetState::Missing,
        );

        assert!(!status.ready);
        assert_eq!(status.status_label, "Download");
        assert!(!status.dataset_ready);
        assert!(status.detail.contains("dataset"));
    }

    #[test]
    fn normalizes_gpqa_dataset_row_from_evalscope_json() {
        let row = gpqa_dataset_row_from_json(
            7,
            &json!({
                "question": "Which energy difference allows two quantum states to be clearly resolved?",
                "choices": ["A. small", "B. medium", "C. large", "D. tiny"],
                "target": "C",
                "metadata": {
                    "correct_answer": "large"
                }
            }),
        );

        assert_eq!(row.index, 7);
        assert!(row.question.contains("energy difference"));
        assert_eq!(row.choices.len(), 4);
        assert_eq!(row.answer.as_deref(), Some("large"));
    }

    #[test]
    fn extracts_gpqa_question_from_evalscope_prompt() {
        let row = gpqa_dataset_row_from_json(
            1,
            &json!({
                "input": [{
                    "content": "Here are some examples...\n\nAnswer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D. Think step by step before answering.\n\nTwo quantum states with energies E1 and E2 have a lifetime of 10^-9 sec and 10^-8 sec, respectively. We want to clearly distinguish these two energy levels. Which one of the following options could be their energy difference so that they can be clearly resolved?\n\n\nA) 10^-11 eV\nB) 10^-9 eV"
                }],
                "choices": ["10^-11 eV", "10^-9 eV"],
                "target": "B"
            }),
        );

        assert!(row.question.starts_with("Two quantum states"));
        assert!(!row.question.contains("Here are some examples"));
        assert!(!row.question.contains("A) 10^-11 eV"));
    }

    #[test]
    fn normalizes_humaneval_dataset_row_from_evalscope_json() {
        let row = humaneval_dataset_row_from_json(
            3,
            &json!({
                "target": "    return value + 1",
                "metadata": {
                    "task_id": "HumanEval/2",
                    "entry_point": "increment",
                    "prompt": "def increment(value: int) -> int:\n    \"\"\"Return value plus one.\"\"\""
                }
            }),
        );

        assert_eq!(row.index, 3);
        assert_eq!(row.task_id, "HumanEval/2");
        assert_eq!(row.entry_point, "increment");
        assert!(row.prompt.starts_with("def increment"));
        assert_eq!(row.canonical_solution, "    return value + 1");
    }

    #[test]
    fn defaults_gpqa_run_config_when_values_are_missing() {
        let config = effective_gpqa_run_config(Some(GpqaRunConfig {
            context_window: None,
            sample_limit: None,
            temperature: None,
            thinking: None,
            top_k: None,
            repeat_penalty: None,
            presence_penalty: None,
            top_p: None,
            min_p: None,
        }))
        .unwrap();

        assert_eq!(
            config,
            EffectiveGpqaRunConfig {
                context_window: 20_000,
                sample_limit: 198,
                temperature: 0.0,
                thinking: GpqaThinkingMode::Off,
                top_k: None,
                repeat_penalty: None,
                presence_penalty: None,
                top_p: None,
                min_p: None,
            }
        );
    }

    #[test]
    fn accepts_gpqa_run_config_within_bounds() {
        let config = effective_gpqa_run_config(Some(GpqaRunConfig {
            context_window: Some(20_000),
            sample_limit: Some(12),
            temperature: Some(0.2),
            thinking: Some(GpqaThinkingMode::On),
            top_k: None,
            repeat_penalty: None,
            presence_penalty: None,
            top_p: None,
            min_p: None,
        }))
        .unwrap();

        assert_eq!(
            config,
            EffectiveGpqaRunConfig {
                context_window: 20_000,
                sample_limit: 12,
                temperature: 0.2,
                thinking: GpqaThinkingMode::On,
                top_k: None,
                repeat_penalty: None,
                presence_penalty: None,
                top_p: None,
                min_p: None,
            }
        );
    }

    #[test]
    fn rejects_gpqa_run_config_outside_bounds() {
        assert!(effective_gpqa_run_config(Some(GpqaRunConfig {
            context_window: Some(0),
            sample_limit: Some(198),
            temperature: Some(0.0),
            thinking: None,
            top_k: None,
            repeat_penalty: None,
            presence_penalty: None,
            top_p: None,
            min_p: None,
        }))
        .is_err());
        assert!(effective_gpqa_run_config(Some(GpqaRunConfig {
            sample_limit: Some(199),
            temperature: Some(0.0),
            context_window: Some(20_000),
            thinking: None,
            top_k: None,
            repeat_penalty: None,
            presence_penalty: None,
            top_p: None,
            min_p: None,
        }))
        .is_err());
        assert!(effective_gpqa_run_config(Some(GpqaRunConfig {
            context_window: Some(20_000),
            sample_limit: Some(198),
            temperature: Some(2.1),
            thinking: None,
            top_k: None,
            repeat_penalty: None,
            presence_penalty: None,
            top_p: None,
            min_p: None,
        }))
        .is_err());
    }

    #[test]
    fn evalscope_generation_config_omits_max_tokens_for_until_eos_generation() {
        let config = EffectiveGpqaRunConfig {
            context_window: 20_000,
            sample_limit: 10,
            temperature: 0.0,
            thinking: GpqaThinkingMode::Off,
            top_k: None,
            repeat_penalty: None,
            presence_penalty: None,
            top_p: None,
            min_p: None,
        };

        let generation_config = gpqa_generation_config(&config);

        assert_eq!(generation_config["temperature"], 0.0);
        assert_eq!(generation_config["stream"], false);
        assert_eq!(
            generation_config["chat_template_kwargs"]["enable_thinking"],
            false
        );
        assert!(generation_config.get("max_tokens").is_none());
        assert!(generation_config.get("max_completion_tokens").is_none());
    }

    #[test]
    fn evalscope_generation_config_includes_sampler_overrides() {
        let config = effective_gpqa_run_config(Some(GpqaRunConfig {
            context_window: Some(20_000),
            sample_limit: Some(10),
            temperature: Some(0.0),
            thinking: Some(GpqaThinkingMode::Off),
            top_k: Some(40),
            repeat_penalty: Some(1.1),
            presence_penalty: Some(0.2),
            top_p: Some(0.95),
            min_p: Some(0.05),
        }))
        .unwrap();

        let generation_config = gpqa_generation_config(&config);

        assert_eq!(generation_config["top_k"], 40);
        assert_eq!(generation_config["repeat_penalty"], 1.1);
        assert_eq!(generation_config["presence_penalty"], 0.2);
        assert_eq!(generation_config["top_p"], 0.95);
        assert_eq!(generation_config["min_p"], 0.05);
    }

    #[test]
    fn evalscope_generation_config_can_enable_template_thinking() {
        let config = EffectiveGpqaRunConfig {
            context_window: 20_000,
            sample_limit: 10,
            temperature: 0.0,
            thinking: GpqaThinkingMode::On,
            top_k: None,
            repeat_penalty: None,
            presence_penalty: None,
            top_p: None,
            min_p: None,
        };

        let generation_config = gpqa_generation_config(&config);

        assert_eq!(
            generation_config["chat_template_kwargs"]["enable_thinking"],
            true
        );
    }

    #[test]
    fn extracts_accuracy_percent_from_evalscope_report() {
        let report = json!({
            "task": "gpqa_diamond",
            "metrics": {
                "mean_acc": 63.1
            },
            "sample_count": 198
        });

        assert_eq!(extract_accuracy_from_json(&report), Some(0.631));
        assert_eq!(
            extract_metric_from_json(&report).as_deref(),
            Some("mean_acc")
        );
        assert_eq!(extract_sample_count_from_json(&report), Some(198));
    }

    #[test]
    fn creates_gpqa_benchmark_result_with_tensor_summary() {
        let report_dir = std::env::temp_dir().join(format!("gpqa-report-test-{}", unix_millis()));
        std::fs::create_dir_all(&report_dir).unwrap();
        let report_path = report_dir.join("gpqa_diamond.json");
        std::fs::write(
            &report_path,
            r#"{"task":"gpqa_diamond","metrics":{"acc":0.5},"sample_count":198}"#,
        )
        .unwrap();

        let tensor_summary = ModelInspectorApiTensorSummary {
            copied_tensor_count: 10,
            converted_tensor_count: 2,
            converted_bytes_before: 120,
            converted_bytes_after: 80,
            requested_target_count: 2,
            verified_target_count: 2,
        };
        let result = gpqa_result_from_report(
            "mock.gguf",
            GpqaShotMode::FiveShotCot,
            &report_path,
            123.0,
            tensor_summary,
            ModelInspectorApiModelSummary {
                tensor_count: 427,
                metadata_count: 33,
            },
            ModelInspectorApiRuntimeSummary {
                prompt_eval_tps: 100.0,
                token_gen_tps: 20.0,
                ttft_ms: 50.0,
                prompt_eval_ms: 10.0,
                generation_ms: 25.0,
                vram_peak_mb: 1024.0,
                vram_allocated_mb: 900.0,
                load_ms: 500.0,
            },
        )
        .unwrap();
        let _ = std::fs::remove_dir_all(&report_dir);

        assert_eq!(result.test_mode, "official_gpqa_diamond");
        assert_eq!(result.converted_tensor_count, 2);
        assert_eq!(result.converted_bytes_before, 120);
        assert_eq!(result.converted_bytes_after, 80);
        assert_eq!(result.requested_target_count, 2);
        assert_eq!(result.verified_target_count, 2);
        assert_eq!(result.model_tensor_count, Some(427));
        assert_eq!(result.model_metadata_count, Some(33));
        assert_eq!(result.token_gen_tps, 20.0);
        assert_eq!(result.ttft_ms, 50.0);
        assert_eq!(result.load_ms, 500.0);
        let standard_eval = result.standard_eval.unwrap();
        assert_eq!(standard_eval.recipe_accuracy, 0.5);
        assert_eq!(standard_eval.tasks[0].metric, "acc");
        assert_eq!(standard_eval.tasks[0].sample_count, 198);
    }
}
