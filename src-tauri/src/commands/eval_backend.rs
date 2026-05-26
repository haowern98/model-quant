use crate::progress::ProgressEmitter;
use std::path::PathBuf;
use std::process::Command;

#[cfg(windows)]
use std::os::windows::process::CommandExt;

const LM_EVAL_PACKAGE: &str = "lm-eval==0.4.12";
const TRANSFORMERS_PACKAGE: &str = "transformers>=4.56,<5";

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OfficialEvalBackendStatus {
    pub installed: bool,
    pub backend_dir: String,
    pub python_path: Option<String>,
    pub lm_eval_available: bool,
    pub adapter_available: bool,
    pub message: String,
}

#[tauri::command]
pub async fn get_official_eval_backend_status() -> Result<OfficialEvalBackendStatus, String> {
    Ok(check_official_eval_backend_status())
}

#[tauri::command]
pub async fn install_official_eval_backend(
    app: tauri::AppHandle,
) -> Result<OfficialEvalBackendStatus, String> {
    tauri::async_runtime::spawn_blocking(move || install_official_eval_backend_blocking(app))
        .await
        .map_err(|err| format!("official eval backend installer failed to run: {}", err))?
}

fn install_official_eval_backend_blocking(
    app: tauri::AppHandle,
) -> Result<OfficialEvalBackendStatus, String> {
    let progress = ProgressEmitter::new(app.clone());
    let backend_dir = official_eval_backend_dir()?;
    progress.installing(0.02, "Preparing official eval backend directory...");
    std::fs::create_dir_all(&backend_dir)
        .map_err(|err| format!("failed to create eval backend directory: {}", err))?;

    let python =
        find_python().ok_or("Python was not found. Install Python 3.10+ and try again.")?;
    let venv_dir = backend_dir.join(".venv");
    if !venv_python(&venv_dir).exists() {
        progress.installing(0.1, "Creating Python virtual environment...");
        run_command(
            Command::new(&python).arg("-m").arg("venv").arg(&venv_dir),
            "failed to create official eval Python venv",
        )?;
    }

    let venv_python = venv_python(&venv_dir);
    progress.installing(0.25, "Upgrading pip inside eval backend...");
    run_command(
        Command::new(&venv_python)
            .arg("-m")
            .arg("pip")
            .arg("install")
            .arg("--upgrade")
            .arg("pip"),
        "failed to upgrade pip in official eval backend",
    )?;
    progress.installing(0.45, "Installing EleutherAI lm-evaluation-harness...");
    run_command(
        Command::new(&venv_python)
            .arg("-m")
            .arg("pip")
            .arg("install")
            .arg("--upgrade")
            .arg(LM_EVAL_PACKAGE)
            .arg(TRANSFORMERS_PACKAGE),
        "failed to install EleutherAI lm-evaluation-harness",
    )?;

    progress.installing(0.9, "Writing Model Surgery lm-eval adapter...");
    write_adapter_package(&backend_dir, &app)?;

    progress.installing(0.96, "Verifying official eval backend...");
    let status = check_official_eval_backend_status();
    if !status.installed {
        return Err(status.message);
    }
    progress.installing(1.0, "Official eval backend is installed.");
    Ok(status)
}

pub fn check_official_eval_backend_status() -> OfficialEvalBackendStatus {
    let backend_dir = match official_eval_backend_dir() {
        Ok(path) => path,
        Err(err) => {
            return OfficialEvalBackendStatus {
                installed: false,
                backend_dir: String::new(),
                python_path: None,
                lm_eval_available: false,
                adapter_available: false,
                message: err,
            };
        }
    };
    let python_path = venv_python(&backend_dir.join(".venv"));
    let lm_eval_available = python_path.exists() && lm_eval_help_available(&python_path);
    let adapter_available = backend_dir
        .join("model_surgery_lm_eval")
        .join("__init__.py")
        .exists();
    let installed = lm_eval_available && adapter_available;
    let message = if installed {
        "Official eval backend is installed".to_string()
    } else if !python_path.exists() {
        "Official eval backend is not installed".to_string()
    } else if !lm_eval_available {
        "Official eval backend venv exists, but lm-eval is not available".to_string()
    } else {
        "Official eval backend adapter is missing".to_string()
    };

    OfficialEvalBackendStatus {
        installed,
        backend_dir: backend_dir.to_string_lossy().into_owned(),
        python_path: python_path
            .exists()
            .then(|| python_path.to_string_lossy().into_owned()),
        lm_eval_available,
        adapter_available,
        message,
    }
}

pub fn official_eval_backend_dir() -> Result<PathBuf, String> {
    if let Some(local_app_data) = std::env::var_os("LOCALAPPDATA") {
        // Keep this path intentionally short: lm-eval ships deeply nested task
        // YAML files that can exceed Windows MAX_PATH under a verbose app dir.
        return Ok(PathBuf::from(local_app_data).join("MSGEval"));
    }

    std::env::current_dir()
        .map(|dir| dir.join(".model-surgery").join("official-eval-backend"))
        .map_err(|err| format!("failed to resolve official eval backend directory: {}", err))
}

fn find_python() -> Option<PathBuf> {
    ["py", "python", "python3"].iter().find_map(|candidate| {
        let mut command = Command::new(candidate);
        configure_command(command.arg("--version"));
        let output = command.output().ok()?;
        output.status.success().then(|| PathBuf::from(candidate))
    })
}

fn lm_eval_help_available(python_path: &std::path::Path) -> bool {
    let mut command = Command::new(python_path);
    configure_command(command.arg("-m").arg("lm_eval").arg("--help"));
    command
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn venv_python(venv_dir: &std::path::Path) -> PathBuf {
    if cfg!(windows) {
        venv_dir.join("Scripts").join("python.exe")
    } else {
        venv_dir.join("bin").join("python")
    }
}

fn run_command(command: &mut Command, context: &str) -> Result<(), String> {
    configure_command(command);
    let output = command
        .output()
        .map_err(|err| format!("{}: {}", context, err))?;
    if output.status.success() {
        return Ok(());
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    Err(format!("{}: {}\n{}", context, stderr.trim(), stdout.trim()))
}

fn configure_command(command: &mut Command) {
    #[cfg(windows)]
    {
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
}

fn write_adapter_package(
    backend_dir: &std::path::Path,
    _app: &tauri::AppHandle,
) -> Result<(), String> {
    let adapter_dir = backend_dir.join("model_surgery_lm_eval");
    std::fs::create_dir_all(&adapter_dir)
        .map_err(|err| format!("failed to create lm-eval adapter directory: {}", err))?;
    std::fs::write(
        adapter_dir.join("__init__.py"),
        "MODEL_SURGERY_LM_EVAL_ADAPTER = True\n",
    )
    .map_err(|err| format!("failed to write lm-eval adapter marker: {}", err))?;
    Ok(())
}
