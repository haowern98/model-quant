use serde::Serialize;
use tauri::{AppHandle, Emitter};

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProgressEvent {
    pub stage: ProgressStage,
    pub percent: f32,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchmarkOutputEvent {
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiOutputEvent {
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub enum ProgressStage {
    #[serde(rename = "requantizing")]
    Requantizing,
    #[serde(rename = "writing")]
    Writing,
    #[serde(rename = "loading")]
    Loading,
    #[serde(rename = "benchmarking")]
    Benchmarking,
}

pub struct ProgressEmitter {
    app: AppHandle,
}

impl ProgressEmitter {
    pub fn new(app: AppHandle) -> Self {
        ProgressEmitter { app }
    }

    pub fn emit(&self, stage: ProgressStage, percent: f32, message: &str) {
        let event = ProgressEvent {
            stage,
            percent: percent.clamp(0.0, 1.0),
            message: message.to_string(),
        };
        let _ = self.app.emit("progress", event);
    }

    pub fn requantizing(&self, percent: f32, info: &str) {
        self.emit(
            ProgressStage::Requantizing,
            percent,
            &format!("Requantizing: {}", info),
        );
    }

    pub fn writing(&self, percent: f32, info: &str) {
        self.emit(
            ProgressStage::Writing,
            percent,
            &format!("Writing: {}", info),
        );
    }

    pub fn loading(&self, percent: f32) {
        self.emit(
            ProgressStage::Loading,
            percent,
            "Loading model into VRAM...",
        );
    }

    pub fn benchmarking(&self, percent: f32) {
        self.emit(
            ProgressStage::Benchmarking,
            percent,
            "Running inference benchmark...",
        );
    }
}

pub fn emit_benchmark_output(app: &AppHandle, message: impl Into<String>) {
    let message = message.into();
    if message.trim().is_empty() {
        return;
    }

    let _ = app.emit("benchmark-output", BenchmarkOutputEvent { message });
}

pub fn emit_api_output(app: &AppHandle, message: impl Into<String>) {
    let message = message.into();
    if message.trim().is_empty() {
        return;
    }

    let _ = app.emit("api-output", ApiOutputEvent { message });
}
