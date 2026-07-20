use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc, Mutex,
};
use std::thread::{self, JoinHandle};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tauri::{AppHandle, State};

use crate::commands::model::{multimodal_projector_path, ProjectorState};
use crate::commands::quant::RecipeStore;
use crate::ffi::runtime_bindings::{
    ChatFinishReason, ChatGenerationParams, MsBaselineBenchmark, MsRuntimeChatSessionCounters,
    RecipeChatSession,
};
use crate::quant::recipe::{QuantType, RecipeState};

const API_CHAT_CONTEXT_TOKENS: u32 = 20_000;
const API_CHAT_UNTIL_CONTEXT_MAX_TOKENS: u32 = 0;
const MULTIMODAL_IMAGE_MARKER: &str = "<__media__>";

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInspectorApiStatus {
    pub running: bool,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub model_id: Option<String>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ModelInspectorApiTensorSummary {
    pub copied_tensor_count: u64,
    pub converted_tensor_count: u64,
    pub converted_bytes_before: u64,
    pub converted_bytes_after: u64,
    pub requested_target_count: u64,
    pub verified_target_count: u64,
}

impl From<MsRuntimeChatSessionCounters> for ModelInspectorApiTensorSummary {
    fn from(counters: MsRuntimeChatSessionCounters) -> Self {
        Self {
            copied_tensor_count: counters.copied_tensor_count,
            converted_tensor_count: counters.converted_tensor_count,
            converted_bytes_before: counters.converted_bytes_before,
            converted_bytes_after: counters.converted_bytes_after,
            requested_target_count: counters.requested_target_count,
            verified_target_count: counters.verified_target_count,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ModelInspectorApiRuntimeSummary {
    pub prompt_eval_tps: f64,
    pub token_gen_tps: f64,
    pub ttft_ms: f64,
    pub prompt_eval_ms: f64,
    pub generation_ms: f64,
    pub vram_peak_mb: f64,
    pub vram_allocated_mb: f64,
    pub load_ms: f64,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ModelInspectorApiModelSummary {
    pub tensor_count: u64,
    pub metadata_count: u64,
}

#[derive(Debug, Default)]
pub(crate) struct ModelInspectorApiRuntimeTotals {
    load_ms: f64,
    completion_count: u64,
    prompt_tokens: u64,
    generated_tokens: u64,
    prompt_eval_ms: f64,
    generation_ms: f64,
    ttft_ms_total: f64,
    vram_peak_mb: f64,
    vram_allocated_mb: f64,
}

impl ModelInspectorApiRuntimeTotals {
    fn new(load_ms: f64) -> Self {
        Self {
            load_ms,
            ..Self::default()
        }
    }

    fn record(&mut self, benchmark: &MsBaselineBenchmark) {
        self.completion_count += 1;
        self.prompt_tokens += benchmark.prompt_tokens as u64;
        self.generated_tokens += benchmark.generated_tokens as u64;
        self.prompt_eval_ms += benchmark.prompt_eval_ms;
        self.generation_ms += benchmark.generation_ms;
        self.ttft_ms_total += benchmark.ttft_ms;
        self.vram_peak_mb = self.vram_peak_mb.max(benchmark.vram_peak_mb);
        self.vram_allocated_mb = self.vram_allocated_mb.max(benchmark.vram_allocated_mb);
    }

    pub(crate) fn snapshot(&self) -> ModelInspectorApiRuntimeSummary {
        ModelInspectorApiRuntimeSummary {
            prompt_eval_tps: tokens_per_second(self.prompt_tokens, self.prompt_eval_ms),
            token_gen_tps: tokens_per_second(self.generated_tokens, self.generation_ms),
            ttft_ms: if self.completion_count == 0 {
                0.0
            } else {
                self.ttft_ms_total / self.completion_count as f64
            },
            prompt_eval_ms: self.prompt_eval_ms,
            generation_ms: self.generation_ms,
            vram_peak_mb: self.vram_peak_mb,
            vram_allocated_mb: self.vram_allocated_mb,
            load_ms: self.load_ms,
        }
    }
}

fn tokens_per_second(tokens: u64, elapsed_ms: f64) -> f64 {
    if elapsed_ms <= 0.0 {
        0.0
    } else {
        tokens as f64 / (elapsed_ms / 1000.0)
    }
}

pub struct ModelInspectorApiState(pub Mutex<ModelInspectorApiLifecycle>);

impl ModelInspectorApiState {
    pub fn new() -> Self {
        Self(Mutex::new(ModelInspectorApiLifecycle::default()))
    }
}

pub struct ModelInspectorApiLifecycle {
    server: Option<ModelInspectorApiServer>,
    startup: Option<ModelInspectorApiStartup>,
}

impl Default for ModelInspectorApiLifecycle {
    fn default() -> Self {
        Self {
            server: None,
            startup: None,
        }
    }
}

#[derive(Clone)]
struct ModelInspectorApiStartup {
    cancelled: Arc<AtomicBool>,
}

impl ModelInspectorApiStartup {
    fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    fn cancel_requested(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    fn same_startup(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.cancelled, &other.cancelled)
    }
}

impl ModelInspectorApiLifecycle {
    fn begin_start(
        &mut self,
    ) -> Result<(ModelInspectorApiStartup, Option<ModelInspectorApiServer>), String> {
        if self.startup.is_some() {
            return Err("ModelInspector API is already starting".to_string());
        }
        let startup = ModelInspectorApiStartup::new();
        self.startup = Some(startup.clone());
        Ok((startup, self.server.take()))
    }

    fn cancel(&mut self) -> Option<ModelInspectorApiServer> {
        if let Some(startup) = self.startup.as_ref() {
            startup.cancel();
        }
        self.server.take()
    }

    fn finish_start(
        &mut self,
        startup: &ModelInspectorApiStartup,
        server: Option<ModelInspectorApiServer>,
    ) -> bool {
        let Some(current) = self.startup.as_ref() else {
            return false;
        };
        if !current.same_startup(startup) {
            return false;
        }
        let cancelled = current.cancel_requested();
        self.startup = None;
        if cancelled {
            return false;
        }
        self.server = server;
        true
    }
}

pub struct ModelInspectorApiServer {
    base_url: String,
    model_id: String,
    token: String,
    tensor_summary: ModelInspectorApiTensorSummary,
    model_summary: ModelInspectorApiModelSummary,
    runtime_totals: Arc<Mutex<ModelInspectorApiRuntimeTotals>>,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl Drop for ModelInspectorApiServer {
    fn drop(&mut self) {
        self.stop();
    }
}

impl ModelInspectorApiServer {
    fn stop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect(
            self.base_url
                .trim_start_matches("http://")
                .trim_end_matches("/v1"),
        );
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

pub fn modelinspector_api_tensor_summary(
    api_state: &ModelInspectorApiState,
    base_url: &str,
    model_id: &str,
) -> Result<ModelInspectorApiTensorSummary, String> {
    let guard = api_state.0.lock().map_err(|e| e.to_string())?;
    Ok(guard
        .server
        .as_ref()
        .filter(|server| server.base_url == base_url && server.model_id == model_id)
        .map(|server| server.tensor_summary)
        .unwrap_or_default())
}

pub(crate) fn modelinspector_api_runtime_totals(
    api_state: &ModelInspectorApiState,
    base_url: &str,
    model_id: &str,
) -> Result<Arc<Mutex<ModelInspectorApiRuntimeTotals>>, String> {
    let guard = api_state.0.lock().map_err(|e| e.to_string())?;
    Ok(guard
        .server
        .as_ref()
        .filter(|server| server.base_url == base_url && server.model_id == model_id)
        .map(|server| server.runtime_totals.clone())
        .unwrap_or_default())
}

pub(crate) fn modelinspector_api_model_summary(
    api_state: &ModelInspectorApiState,
    base_url: &str,
    model_id: &str,
) -> Result<ModelInspectorApiModelSummary, String> {
    let guard = api_state.0.lock().map_err(|e| e.to_string())?;
    Ok(guard
        .server
        .as_ref()
        .filter(|server| server.base_url == base_url && server.model_id == model_id)
        .map(|server| server.model_summary)
        .unwrap_or_default())
}

#[tauri::command]
pub async fn start_modelinspector_api(
    benchmark_label: Option<String>,
    benchmark_sample_count: Option<u64>,
    context_window: Option<u32>,
    default_enable_thinking: Option<bool>,
    enable_multimodal: Option<bool>,
    app: AppHandle,
    api_state: State<'_, ModelInspectorApiState>,
    recipe_state: State<'_, RecipeStore>,
    projector_state: State<'_, ProjectorState>,
) -> Result<ModelInspectorApiStatus, String> {
    let recipe = recipe_state
        .0
        .lock()
        .map_err(|e| e.to_string())?
        .clone()
        .ok_or("No model loaded")?;

    let projector_path = if enable_multimodal.unwrap_or(false) {
        Some(multimodal_projector_path(
            &recipe.base_model,
            &projector_state,
        )?)
    } else {
        None
    };

    let gguf_summary = crate::ffi::runtime_bindings::inspect_gguf(&recipe.base_model)?;
    let model_summary = ModelInspectorApiModelSummary {
        tensor_count: gguf_summary.tensor_count,
        metadata_count: gguf_summary.metadata_count,
    };
    let model_id = model_id_from_path(&recipe.base_model);
    let token = make_token();
    let (startup, old_server) = {
        let mut guard = api_state.0.lock().map_err(|e| e.to_string())?;
        guard.begin_start()?
    };
    if let Some(mut server) = old_server {
        server.stop();
    }
    crate::ffi::runtime_bindings::reset_recipe_test_cancel();

    let listener = match TcpListener::bind("127.0.0.1:0") {
        Ok(listener) => listener,
        Err(error) => {
            let mut guard = api_state.0.lock().map_err(|e| e.to_string())?;
            guard.finish_start(&startup, None);
            return Err(format!("Failed to bind Model Inspector API: {error}"));
        }
    };
    let addr = match listener.local_addr() {
        Ok(addr) => addr,
        Err(error) => {
            let mut guard = api_state.0.lock().map_err(|e| e.to_string())?;
            guard.finish_start(&startup, None);
            return Err(format!(
                "Failed to read Model Inspector API address: {error}"
            ));
        }
    };
    let base_url = format!("http://{addr}/v1");
    let stop = Arc::new(AtomicBool::new(false));
    let targets = recipe_targets(&recipe);
    crate::progress::emit_api_output(&app, "ModelInspector API: loading in-process model session");
    let output_app = app.clone();
    let context_tokens = context_window.unwrap_or(API_CHAT_CONTEXT_TOKENS);
    if context_tokens == 0 {
        return Err("ModelInspector API context window must be greater than 0.".to_string());
    }
    let load_start = Instant::now();
    let session =
        match crate::ffi::runtime_bindings::open_recipe_chat_session_with_projector_and_progress(
            &recipe.base_model,
            projector_path.as_deref(),
            &targets,
            context_tokens,
            |message| {
                crate::progress::emit_api_output(&output_app, message);
            },
        ) {
            Ok(session) => session,
            Err(error) => {
                let mut guard = api_state.0.lock().map_err(|e| e.to_string())?;
                let was_cancelled = startup.cancel_requested();
                guard.finish_start(&startup, None);
                if was_cancelled || error.to_lowercase().contains("cancel") {
                    return Err("ModelInspector API startup cancelled".to_string());
                }
                return Err(format!(
                    "Failed to load Model Inspector API model session: {error}"
                ));
            }
        };
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    if startup.cancel_requested() {
        let mut guard = api_state.0.lock().map_err(|e| e.to_string())?;
        guard.finish_start(&startup, None);
        return Err("ModelInspector API startup cancelled".to_string());
    }
    let tensor_summary = session
        .counters()
        .map(ModelInspectorApiTensorSummary::from)
        .unwrap_or_default();
    for line in [
        format!(
            "ModelInspector API summary: copied tensors {}",
            tensor_summary.copied_tensor_count
        ),
        format!(
            "ModelInspector API summary: converted tensors {}",
            tensor_summary.converted_tensor_count
        ),
        format!(
            "ModelInspector API summary: converted from {} B",
            tensor_summary.converted_bytes_before
        ),
        format!(
            "ModelInspector API summary: converted to {} B",
            tensor_summary.converted_bytes_after
        ),
        format!(
            "ModelInspector API summary: verified targets {}/{}",
            tensor_summary.verified_target_count, tensor_summary.requested_target_count
        ),
    ] {
        crate::progress::emit_api_output(&app, line);
    }
    let runtime_totals = Arc::new(Mutex::new(ModelInspectorApiRuntimeTotals::new(load_ms)));
    let server_state = Arc::new(HttpApiState {
        token: token.clone(),
        model_id: model_id.clone(),
        session: Some(Mutex::new(session)),
        runtime_totals: runtime_totals.clone(),
        benchmark_label,
        benchmark_sample_count,
        default_enable_thinking,
        vision_enabled: projector_path.is_some(),
        completion_count: AtomicU64::new(0),
        app: Some(app),
    });

    let thread_stop = stop.clone();
    let thread_state = server_state.clone();
    let handle = match thread::Builder::new()
        .name("modelinspector-api".to_string())
        .spawn(move || run_server(listener, thread_stop, thread_state))
    {
        Ok(handle) => handle,
        Err(error) => {
            let mut guard = api_state.0.lock().map_err(|e| e.to_string())?;
            guard.finish_start(&startup, None);
            return Err(format!(
                "Failed to start Model Inspector API thread: {error}"
            ));
        }
    };

    let server = ModelInspectorApiServer {
        base_url: base_url.clone(),
        model_id: model_id.clone(),
        token,
        tensor_summary,
        model_summary,
        runtime_totals,
        stop,
        handle: Some(handle),
    };
    let api_key = server.token.clone();
    let mut guard = api_state.0.lock().map_err(|e| e.to_string())?;
    if !guard.finish_start(&startup, Some(server)) {
        return Err("ModelInspector API startup cancelled".to_string());
    }
    if let Some(app) = server_state.app.as_ref() {
        crate::progress::emit_api_output(app, format!("ModelInspector API ready at {base_url}"));
    }

    Ok(ModelInspectorApiStatus {
        running: true,
        base_url: Some(base_url),
        api_key: Some(api_key),
        model_id: Some(model_id),
    })
}

#[tauri::command]
pub async fn stop_modelinspector_api(
    api_state: State<'_, ModelInspectorApiState>,
) -> Result<ModelInspectorApiStatus, String> {
    crate::ffi::runtime_bindings::cancel_recipe_test();
    let server = {
        let mut guard = api_state.0.lock().map_err(|e| e.to_string())?;
        guard.cancel()
    };
    if let Some(mut server) = server {
        server.stop();
    }
    Ok(ModelInspectorApiStatus {
        running: false,
        base_url: None,
        api_key: None,
        model_id: None,
    })
}

#[tauri::command]
pub async fn get_modelinspector_api_status(
    api_state: State<'_, ModelInspectorApiState>,
) -> Result<ModelInspectorApiStatus, String> {
    let guard = api_state.0.lock().map_err(|e| e.to_string())?;
    Ok(match guard.server.as_ref() {
        Some(server) => ModelInspectorApiStatus {
            running: true,
            base_url: Some(server.base_url.clone()),
            api_key: Some(server.token.clone()),
            model_id: Some(server.model_id.clone()),
        },
        None => ModelInspectorApiStatus {
            running: false,
            base_url: None,
            api_key: None,
            model_id: None,
        },
    })
}

#[derive(Debug)]
struct HttpApiState {
    token: String,
    model_id: String,
    session: Option<Mutex<RecipeChatSession>>,
    runtime_totals: Arc<Mutex<ModelInspectorApiRuntimeTotals>>,
    benchmark_label: Option<String>,
    benchmark_sample_count: Option<u64>,
    default_enable_thinking: Option<bool>,
    vision_enabled: bool,
    completion_count: AtomicU64,
    app: Option<AppHandle>,
}

#[derive(Debug)]
struct HttpRequest {
    method: String,
    path: String,
    authorization: Option<String>,
    body: String,
}

#[derive(Debug)]
struct HttpResponse {
    status: u16,
    reason: &'static str,
    body: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    model: Option<String>,
    messages: Option<Vec<ChatMessage>>,
    stream: Option<bool>,
    max_tokens: Option<u32>,
    max_completion_tokens: Option<u32>,
    n_predict: Option<i32>,
    stop: Option<StopField>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<i32>,
    min_p: Option<f64>,
    typical_p: Option<f64>,
    repeat_last_n: Option<i32>,
    repeat_penalty: Option<f64>,
    frequency_penalty: Option<f64>,
    presence_penalty: Option<f64>,
    dry_multiplier: Option<f64>,
    dry_base: Option<f64>,
    dry_allowed_length: Option<i32>,
    dry_penalty_last_n: Option<i32>,
    seed: Option<i64>,
    add_generation_prompt: Option<bool>,
    chat_template_kwargs: Option<Value>,
    reasoning_format: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    role: String,
    content: Value,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum StopField {
    One(String),
    Many(Vec<String>),
}

#[derive(Debug, Clone)]
struct ChatGenerationConfig {
    stop: Vec<String>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<i32>,
    min_p: Option<f64>,
    typical_p: Option<f64>,
    repeat_last_n: Option<i32>,
    repeat_penalty: Option<f64>,
    frequency_penalty: Option<f64>,
    presence_penalty: Option<f64>,
    dry_multiplier: Option<f64>,
    dry_base: Option<f64>,
    dry_allowed_length: Option<i32>,
    dry_penalty_last_n: Option<i32>,
    seed: Option<i64>,
    add_generation_prompt: Option<bool>,
    chat_template_kwargs: Option<Value>,
    thinking_override: Option<ChatTemplateThinkingOverride>,
    reasoning_format: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ChatTemplateThinkingOverride {
    configured: bool,
    requested: bool,
}

fn chat_template_kwargs_with_thinking_default(
    request_kwargs: Option<&Value>,
    default_enable_thinking: Option<bool>,
) -> (Option<Value>, Option<ChatTemplateThinkingOverride>) {
    let configured = default_enable_thinking.unwrap_or(false);
    match request_kwargs {
        Some(Value::Object(values)) => {
            let mut values = values.clone();
            let thinking_override = match values.get("enable_thinking") {
                Some(value) => value.as_bool().and_then(|requested| {
                    (requested != configured).then_some(ChatTemplateThinkingOverride {
                        configured,
                        requested,
                    })
                }),
                None => {
                    values.insert("enable_thinking".to_string(), Value::Bool(configured));
                    None
                }
            };
            (Some(Value::Object(values)), thinking_override)
        }
        Some(value) => (Some(value.clone()), None),
        None => {
            let mut values = serde_json::Map::new();
            values.insert("enable_thinking".to_string(), Value::Bool(configured));
            (Some(Value::Object(values)), None)
        }
    }
}

fn chat_generation_config(
    payload: &ChatCompletionRequest,
    default_enable_thinking: Option<bool>,
) -> ChatGenerationConfig {
    let _accepted_but_ignored_token_limits = (
        payload.max_tokens,
        payload.max_completion_tokens,
        payload.n_predict,
    );
    let (chat_template_kwargs, thinking_override) = chat_template_kwargs_with_thinking_default(
        payload.chat_template_kwargs.as_ref(),
        default_enable_thinking,
    );
    ChatGenerationConfig {
        stop: stop_strings(payload.stop.as_ref()),
        temperature: payload.temperature,
        top_p: payload.top_p,
        top_k: payload.top_k,
        min_p: payload.min_p,
        typical_p: payload.typical_p,
        repeat_last_n: payload.repeat_last_n,
        repeat_penalty: payload.repeat_penalty,
        frequency_penalty: payload.frequency_penalty,
        presence_penalty: payload.presence_penalty,
        dry_multiplier: payload.dry_multiplier,
        dry_base: payload.dry_base,
        dry_allowed_length: payload.dry_allowed_length,
        dry_penalty_last_n: payload.dry_penalty_last_n,
        seed: payload.seed,
        add_generation_prompt: payload.add_generation_prompt,
        chat_template_kwargs,
        thinking_override,
        reasoning_format: payload.reasoning_format.clone(),
    }
}

impl ChatGenerationConfig {
    fn to_native_params(&self) -> ChatGenerationParams {
        let defaults = ChatGenerationParams::default();
        ChatGenerationParams {
            max_tokens: API_CHAT_UNTIL_CONTEXT_MAX_TOKENS,
            add_generation_prompt: u32::from(self.add_generation_prompt.unwrap_or(true)),
            seed: self
                .seed
                .and_then(|value| u32::try_from(value).ok())
                .unwrap_or(defaults.seed),
            top_k: self.top_k.unwrap_or(defaults.top_k),
            repeat_last_n: self.repeat_last_n.unwrap_or(defaults.repeat_last_n),
            dry_allowed_length: self
                .dry_allowed_length
                .unwrap_or(defaults.dry_allowed_length),
            dry_penalty_last_n: self
                .dry_penalty_last_n
                .unwrap_or(defaults.dry_penalty_last_n),
            temperature: self.temperature.unwrap_or(defaults.temperature),
            top_p: self.top_p.unwrap_or(defaults.top_p),
            min_p: self.min_p.unwrap_or(defaults.min_p),
            typical_p: self.typical_p.unwrap_or(defaults.typical_p),
            repeat_penalty: self.repeat_penalty.unwrap_or(defaults.repeat_penalty),
            frequency_penalty: self.frequency_penalty.unwrap_or(defaults.frequency_penalty),
            presence_penalty: self.presence_penalty.unwrap_or(defaults.presence_penalty),
            dry_multiplier: self.dry_multiplier.unwrap_or(defaults.dry_multiplier),
            dry_base: self.dry_base.unwrap_or(defaults.dry_base),
        }
    }

    fn native_stop_strings(&self) -> Vec<String> {
        self.stop
            .iter()
            .filter(|stop| !stop.is_empty())
            .cloned()
            .collect()
    }

    fn native_chat_template_kwargs_json(&self) -> Option<String> {
        self.chat_template_kwargs
            .as_ref()
            .map(serde_json::to_string)
            .transpose()
            .ok()
            .flatten()
    }

    fn native_reasoning_format(&self) -> Option<&str> {
        self.reasoning_format.as_deref()
    }
}

fn stop_strings(stop: Option<&StopField>) -> Vec<String> {
    match stop {
        Some(StopField::One(value)) => vec![value.clone()],
        Some(StopField::Many(values)) => values.clone(),
        None => Vec::new(),
    }
}

fn chat_messages_for_generation(
    messages: Option<&[ChatMessage]>,
) -> Result<ChatGenerationMessages, String> {
    let messages = messages.unwrap_or_default();
    let mut native_messages = Vec::with_capacity(messages.len());
    let mut image_data_urls = Vec::new();
    for message in messages {
        let content = chat_message_content_for_generation(&message.content, &mut image_data_urls)?;
        native_messages.push((message.role.clone(), content));
    }
    Ok(ChatGenerationMessages {
        messages: native_messages,
        image_data_urls,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ChatGenerationMessages {
    messages: Vec<(String, String)>,
    image_data_urls: Vec<String>,
}

fn chat_message_content_for_generation(
    content: &Value,
    image_data_urls: &mut Vec<String>,
) -> Result<String, String> {
    match content {
        Value::String(text) => Ok(text.clone()),
        Value::Array(parts) => {
            let mut text = String::new();
            for part in parts {
                let part_type = part.get("type").and_then(Value::as_str);
                match part_type {
                    Some("text") => {
                        text.push_str(part.get("text").and_then(Value::as_str).unwrap_or_default());
                    }
                    Some("image_url") => {
                        let image_url = part
                            .get("image_url")
                            .and_then(|value| {
                                value
                                    .as_str()
                                    .or_else(|| value.get("url").and_then(Value::as_str))
                            })
                            .ok_or("image_url content part requires a string url")?;
                        if !image_url.starts_with("data:image/") {
                            return Err(
                                "image_url must be a base64 data:image URL; remote URLs are not supported"
                                    .to_string(),
                            );
                        }
                        text.push_str(MULTIMODAL_IMAGE_MARKER);
                        image_data_urls.push(image_url.to_string());
                    }
                    _ => {
                        return Err(
                            "messages must use string content, text content parts, or image_url data:image parts"
                                .to_string(),
                        )
                    }
                }
            }
            Ok(text)
        }
        _ => Err(
            "messages must use string content, text content parts, or image_url data:image parts"
                .to_string(),
        ),
    }
}

fn run_server(listener: TcpListener, stop: Arc<AtomicBool>, state: Arc<HttpApiState>) {
    for stream in listener.incoming() {
        if stop.load(Ordering::SeqCst) {
            break;
        }

        match stream {
            Ok(mut stream) => {
                let state = state.clone();
                let _ = handle_connection(&mut stream, &state);
            }
            Err(_) if stop.load(Ordering::SeqCst) => break,
            Err(_) => continue,
        }
    }
}

fn handle_connection(stream: &mut TcpStream, state: &HttpApiState) -> Result<(), String> {
    let mut buffer = Vec::new();
    let mut chunk = [0_u8; 4096];
    loop {
        let read = stream.read(&mut chunk).map_err(|e| e.to_string())?;
        if read == 0 {
            break;
        }
        buffer.extend_from_slice(&chunk[..read]);
        if request_complete(&buffer) {
            break;
        }
    }

    let raw = String::from_utf8_lossy(&buffer);
    let request = parse_request(&raw)?;
    if request.method == "OPTIONS" {
        return write_response(stream, json_response(204, "No Content", json!({})));
    }
    if !is_public_route(&request) && !authorized(&request, &state.token) {
        return write_response(
            stream,
            json_response(
                401,
                "Unauthorized",
                json!({
                    "error": {
                        "message": "Missing or invalid bearer token",
                        "type": "authentication_error"
                    }
                }),
            ),
        );
    }
    if request.method == "POST"
        && request.path == "/v1/chat/completions"
        && request_wants_streaming_chat_completion(&request)
    {
        return chat_completions_stream(stream, &request, state);
    }

    let response = handle_request(&request, state);
    write_response(stream, response)
}

fn request_complete(buffer: &[u8]) -> bool {
    let Some(header_end) = find_header_end(buffer) else {
        return false;
    };
    let headers = String::from_utf8_lossy(&buffer[..header_end]);
    let content_length = headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .unwrap_or(0);
    buffer.len() >= header_end + 4 + content_length
}

fn find_header_end(buffer: &[u8]) -> Option<usize> {
    buffer.windows(4).position(|window| window == b"\r\n\r\n")
}

fn parse_request(raw: &str) -> Result<HttpRequest, String> {
    let (head, body) = raw
        .split_once("\r\n\r\n")
        .ok_or("Malformed HTTP request: missing header terminator")?;
    let mut lines = head.lines();
    let request_line = lines
        .next()
        .ok_or("Malformed HTTP request: missing request line")?;
    let mut request_parts = request_line.split_whitespace();
    let method = request_parts
        .next()
        .ok_or("Malformed HTTP request: missing method")?
        .to_string();
    let path = request_parts
        .next()
        .ok_or("Malformed HTTP request: missing path")?
        .to_string();
    let mut authorization = None;
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            if name.eq_ignore_ascii_case("authorization") {
                authorization = Some(value.trim().to_string());
            }
        }
    }
    Ok(HttpRequest {
        method,
        path,
        authorization,
        body: body.to_string(),
    })
}

fn handle_request(request: &HttpRequest, state: &HttpApiState) -> HttpResponse {
    if !is_public_route(request) && !authorized(request, &state.token) {
        return json_response(
            401,
            "Unauthorized",
            json!({
                "error": {
                    "message": "Missing or invalid bearer token",
                    "type": "authentication_error"
                }
            }),
        );
    }

    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/v1/models") => json_response(
            200,
            "OK",
            json!({
                "object": "list",
                "data": [{
                    "id": state.model_id,
                    "object": "model",
                    "owned_by": "modelinspector"
                }]
            }),
        ),
        ("POST", "/v1/chat/completions") => chat_completions(request, state),
        _ => json_response(
            404,
            "Not Found",
            json!({
                "error": {
                    "message": "Unsupported Model Inspector API route",
                    "type": "invalid_request_error"
                }
            }),
        ),
    }
}

fn is_public_route(request: &HttpRequest) -> bool {
    matches!(
        (request.method.as_str(), request.path.as_str()),
        ("GET", "/v1/models")
    )
}

fn authorized(request: &HttpRequest, token: &str) -> bool {
    request
        .authorization
        .as_deref()
        .map(|value| value == format!("Bearer {token}"))
        .unwrap_or(false)
}

fn request_wants_streaming_chat_completion(request: &HttpRequest) -> bool {
    serde_json::from_str::<ChatCompletionRequest>(&request.body)
        .map(|payload| payload.stream.unwrap_or(false))
        .unwrap_or(false)
}

fn chat_completions(request: &HttpRequest, state: &HttpApiState) -> HttpResponse {
    let payload = match serde_json::from_str::<ChatCompletionRequest>(&request.body) {
        Ok(payload) => payload,
        Err(error) => {
            return json_response(
                400,
                "Bad Request",
                json!({
                    "error": {
                        "message": format!("Invalid chat completion request: {error}"),
                        "type": "invalid_request_error"
                    }
                }),
            )
        }
    };

    if payload.stream.unwrap_or(false) {
        return json_response(
            400,
            "Bad Request",
            json!({
                "error": {
                    "message": "Streaming chat completions are not supported yet",
                    "type": "invalid_request_error"
                }
            }),
        );
    }

    let config = chat_generation_config(&payload, state.default_enable_thinking);
    emit_thinking_override(state, config.thinking_override);
    let messages = match chat_messages_for_generation(payload.messages.as_deref()) {
        Ok(messages) => messages,
        Err(message) => {
            return json_response(
                400,
                "Bad Request",
                json!({
                    "error": {
                        "message": message,
                        "type": "invalid_request_error"
                    }
                }),
            );
        }
    };
    let native_params = config.to_native_params();
    let model = payload.model.unwrap_or_else(|| state.model_id.clone());
    if messages.messages.is_empty()
        || messages
            .messages
            .iter()
            .all(|(_, content)| content.trim().is_empty())
    {
        return json_response(
            400,
            "Bad Request",
            json!({
                "error": {
                    "message": "Chat completion request must include at least one message",
                    "type": "invalid_request_error"
                }
            }),
        );
    }

    if !messages.image_data_urls.is_empty() && !state.vision_enabled {
        return json_response(
            400,
            "Bad Request",
            json!({
                "error": {
                    "message": "Image input requires an MMPROJ GGUF loaded in the MMPROJ section.",
                    "type": "invalid_request_error"
                }
            }),
        );
    }

    let completion_count = prospective_completion_count(state);
    let (mut content, reasoning_content, benchmark, mut finish_reason) = match state
        .session
        .as_ref()
    {
        Some(session) => {
            let app = state.app.clone();
            let benchmark_label = state.benchmark_label.clone();
            let benchmark_sample_count = state.benchmark_sample_count;
            match session
                .lock()
                .map_err(|e| e.to_string())
                .and_then(|mut session| {
                    if messages.image_data_urls.is_empty() {
                        session.generate_chat_streaming(
                            &messages.messages,
                            &native_params,
                            &config.native_stop_strings(),
                            config.native_chat_template_kwargs_json().as_deref(),
                            config.native_reasoning_format(),
                            |visible_delta, reasoning_delta| {
                                emit_streaming_completion_delta(
                                    app.as_ref(),
                                    benchmark_label.as_deref(),
                                    completion_count,
                                    benchmark_sample_count,
                                    visible_delta,
                                    reasoning_delta,
                                );
                                Ok(())
                            },
                        )
                    } else {
                        session.generate_chat_multimodal_streaming(
                            &messages.messages,
                            &messages.image_data_urls,
                            &native_params,
                            &config.native_stop_strings(),
                            config.native_chat_template_kwargs_json().as_deref(),
                            config.native_reasoning_format(),
                            |visible_delta, reasoning_delta| {
                                emit_streaming_completion_delta(
                                    app.as_ref(),
                                    benchmark_label.as_deref(),
                                    completion_count,
                                    benchmark_sample_count,
                                    visible_delta,
                                    reasoning_delta,
                                );
                                Ok(())
                            },
                        )
                    }
                }) {
                Ok(output) => (
                    output.text,
                    output.reasoning_text,
                    output.benchmark,
                    chat_finish_reason_label(output.finish_reason),
                ),
                Err(error) => {
                    return json_response(
                        500,
                        "Internal Server Error",
                        json!({
                            "error": {
                                "message": format!("Native generation failed: {error}"),
                                "type": "server_error"
                            }
                        }),
                    )
                }
            }
        }
        None => {
            let prompt_chars = messages
                .messages
                .iter()
                .map(|(_, content)| content.chars().count())
                .sum::<usize>();
            (
                format!(
                    "Model Inspector API smoke response for {model}. Received {prompt_chars} prompt character(s)."
                ),
                None,
                empty_benchmark(),
                "stop",
            )
        }
    };
    if state.session.is_none() {
        finish_reason = apply_stop_strings(&mut content, &config.stop);
    }
    record_completion_benchmark(state, &benchmark);
    let diagnostics = ChatCompletionDiagnostics {
        model: model.clone(),
        finish_reason,
        prompt_tokens: benchmark.prompt_tokens,
        completion_tokens: benchmark.generated_tokens,
        total_tokens: benchmark.prompt_tokens + benchmark.generated_tokens,
        visible_content: content.clone(),
        reasoning_content: reasoning_content.clone(),
    };
    emit_completion_output(state, completion_count, &diagnostics);

    json_response(200, "OK", {
        let assistant_message = assistant_message_json(&content, reasoning_content.as_deref());
        json!({
        "id": format!("chatcmpl-{}", unix_millis()),
        "object": "chat.completion",
        "created": unix_seconds(),
        "model": model,
        "choices": [{
            "index": 0,
            "message": assistant_message,
            "finish_reason": finish_reason
        }],
        "usage": {
            "prompt_tokens": benchmark.prompt_tokens,
            "completion_tokens": benchmark.generated_tokens,
            "total_tokens": benchmark.prompt_tokens + benchmark.generated_tokens
        }
        })
    })
}

fn chat_completions_stream(
    stream: &mut TcpStream,
    request: &HttpRequest,
    state: &HttpApiState,
) -> Result<(), String> {
    let payload = match serde_json::from_str::<ChatCompletionRequest>(&request.body) {
        Ok(payload) => payload,
        Err(error) => {
            return write_response(
                stream,
                json_response(
                    400,
                    "Bad Request",
                    json!({
                        "error": {
                            "message": format!("Invalid chat completion request: {error}"),
                            "type": "invalid_request_error"
                        }
                    }),
                ),
            )
        }
    };

    let config = chat_generation_config(&payload, state.default_enable_thinking);
    emit_thinking_override(state, config.thinking_override);
    let messages = match chat_messages_for_generation(payload.messages.as_deref()) {
        Ok(messages) => messages,
        Err(message) => {
            return write_response(
                stream,
                json_response(
                    400,
                    "Bad Request",
                    json!({
                        "error": {
                            "message": message,
                            "type": "invalid_request_error"
                        }
                    }),
                ),
            );
        }
    };
    let native_params = config.to_native_params();
    let model = payload.model.unwrap_or_else(|| state.model_id.clone());
    if messages.messages.is_empty()
        || messages
            .messages
            .iter()
            .all(|(_, content)| content.trim().is_empty())
    {
        return write_response(
            stream,
            json_response(
                400,
                "Bad Request",
                json!({
                    "error": {
                        "message": "Chat completion request must include at least one message",
                        "type": "invalid_request_error"
                    }
                }),
            ),
        );
    }

    if !messages.image_data_urls.is_empty() && !state.vision_enabled {
        return write_response(
            stream,
            json_response(
                400,
                "Bad Request",
                json!({
                    "error": {
                        "message": "Image input requires an MMPROJ GGUF loaded in the MMPROJ section.",
                        "type": "invalid_request_error"
                    }
                }),
            ),
        );
    }

    let id = format!("chatcmpl-{}", unix_millis());
    let created = unix_seconds();
    let completion_count = prospective_completion_count(state);
    let final_result = match state.session.as_ref() {
        Some(session) => {
            let mut session = match session.lock().map_err(|e| e.to_string()) {
                Ok(session) => session,
                Err(error) => {
                    return write_response(
                        stream,
                        json_response(
                            500,
                            "Internal Server Error",
                            json!({
                                "error": {
                                    "message": format!("Native generation failed: {error}"),
                                    "type": "server_error"
                                }
                            }),
                        ),
                    )
                }
            };
            write_sse_headers(stream)?;
            let app = state.app.clone();
            let benchmark_label = state.benchmark_label.clone();
            let benchmark_sample_count = state.benchmark_sample_count;
            let generation = if messages.image_data_urls.is_empty() {
                session.generate_chat_streaming(
                    &messages.messages,
                    &native_params,
                    &config.native_stop_strings(),
                    config.native_chat_template_kwargs_json().as_deref(),
                    config.native_reasoning_format(),
                    |visible_delta, reasoning_delta| {
                        emit_streaming_completion_delta(
                            app.as_ref(),
                            benchmark_label.as_deref(),
                            completion_count,
                            benchmark_sample_count,
                            visible_delta,
                            reasoning_delta,
                        );
                        write_chat_completion_stream_delta(
                            stream,
                            &id,
                            created,
                            &model,
                            visible_delta,
                            reasoning_delta,
                        )
                    },
                )
            } else {
                session.generate_chat_multimodal_streaming(
                    &messages.messages,
                    &messages.image_data_urls,
                    &native_params,
                    &config.native_stop_strings(),
                    config.native_chat_template_kwargs_json().as_deref(),
                    config.native_reasoning_format(),
                    |visible_delta, reasoning_delta| {
                        emit_streaming_completion_delta(
                            app.as_ref(),
                            benchmark_label.as_deref(),
                            completion_count,
                            benchmark_sample_count,
                            visible_delta,
                            reasoning_delta,
                        );
                        write_chat_completion_stream_delta(
                            stream,
                            &id,
                            created,
                            &model,
                            visible_delta,
                            reasoning_delta,
                        )
                    },
                )
            };
            match generation {
                Ok(output) => Ok((
                    output.text,
                    output.reasoning_text,
                    output.benchmark,
                    chat_finish_reason_label(output.finish_reason),
                )),
                Err(error) => Err(error),
            }
        }
        None => {
            write_sse_headers(stream)?;
            let prompt_chars = messages
                .messages
                .iter()
                .map(|(_, content)| content.chars().count())
                .sum::<usize>();
            let mut content = format!(
                "Model Inspector API smoke response for {model}. Received {prompt_chars} prompt character(s)."
            );
            let finish_reason = apply_stop_strings(&mut content, &config.stop);
            write_chat_completion_stream_delta(stream, &id, created, &model, &content, "")?;
            Ok((content, None, empty_benchmark(), finish_reason))
        }
    };

    match final_result {
        Ok((content, reasoning_content, benchmark, finish_reason)) => {
            record_completion_benchmark(state, &benchmark);
            let diagnostics = ChatCompletionDiagnostics {
                model: model.clone(),
                finish_reason,
                prompt_tokens: benchmark.prompt_tokens,
                completion_tokens: benchmark.generated_tokens,
                total_tokens: benchmark.prompt_tokens + benchmark.generated_tokens,
                visible_content: content,
                reasoning_content,
            };
            emit_completion_output(state, completion_count, &diagnostics);
            write_chat_completion_stream_finish(
                stream,
                &id,
                created,
                &model,
                finish_reason,
                &benchmark,
            )?;
            write_sse_done(stream)
        }
        Err(error) => {
            write_sse_data(
                stream,
                &json!({
                    "error": {
                        "message": format!("Native generation failed: {error}"),
                        "type": "server_error"
                    }
                }),
            )?;
            write_sse_done(stream)
        }
    }
}

#[derive(Debug, Clone)]
struct ChatCompletionDiagnostics {
    model: String,
    finish_reason: &'static str,
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
    visible_content: String,
    reasoning_content: Option<String>,
}

fn next_completion_count(state: &HttpApiState) -> u64 {
    state.completion_count.fetch_add(1, Ordering::SeqCst) + 1
}

fn prospective_completion_count(state: &HttpApiState) -> u64 {
    state.completion_count.load(Ordering::SeqCst) + 1
}

fn record_completion_benchmark(state: &HttpApiState, benchmark: &MsBaselineBenchmark) {
    if let Ok(mut totals) = state.runtime_totals.lock() {
        totals.record(benchmark);
    }
}

fn emit_completion_output(
    state: &HttpApiState,
    expected_count: u64,
    diagnostics: &ChatCompletionDiagnostics,
) {
    let Some(label) = state.benchmark_label.as_deref() else {
        return;
    };
    let count = next_completion_count(state);
    debug_assert_eq!(count, expected_count);
    if let Some(app) = state.app.as_ref() {
        crate::progress::emit_api_output(
            app,
            benchmark_completion_output(label, count, state.benchmark_sample_count, diagnostics),
        );
    }
}

fn emit_streaming_completion_delta(
    app: Option<&AppHandle>,
    label: Option<&str>,
    count: u64,
    total: Option<u64>,
    visible_delta: &str,
    reasoning_delta: &str,
) {
    let (Some(app), Some(label)) = (app, label) else {
        return;
    };
    for message in
        streaming_completion_delta_outputs(label, count, total, visible_delta, reasoning_delta)
    {
        crate::progress::emit_api_output(app, message);
    }
}

fn streaming_completion_delta_outputs(
    label: &str,
    count: u64,
    total: Option<u64>,
    visible_delta: &str,
    reasoning_delta: &str,
) -> Vec<String> {
    let progress = total
        .map(|total| format!(" {count}/{total}"))
        .unwrap_or_else(|| format!(" {count}"));
    let mut output = Vec::new();
    if !reasoning_delta.is_empty() {
        output.push(format!(
            "{label}: chat completion request{progress} reasoning delta\n{reasoning_delta}"
        ));
    }
    if !visible_delta.is_empty() {
        output.push(format!(
            "{label}: chat completion request{progress} visible delta\n{visible_delta}"
        ));
    }
    output
}

fn emit_thinking_override(
    state: &HttpApiState,
    thinking_override: Option<ChatTemplateThinkingOverride>,
) {
    let Some(thinking_override) = thinking_override else {
        return;
    };
    let (Some(label), Some(app)) = (state.benchmark_label.as_deref(), state.app.as_ref()) else {
        return;
    };
    crate::progress::emit_api_output(
        app,
        &format!(
            "{label}: request chat_template_kwargs.enable_thinking={} overrides configured default={}",
            thinking_override.requested, thinking_override.configured
        ),
    );
}

fn assistant_message_json(content: &str, reasoning_content: Option<&str>) -> Value {
    let mut message = json!({
        "role": "assistant",
        "content": content
    });
    if let Some(reasoning) = reasoning_content.filter(|value| !value.is_empty()) {
        message["reasoning_content"] = json!(reasoning);
    }
    message
}

fn benchmark_completion_output(
    label: &str,
    count: u64,
    total: Option<u64>,
    diagnostics: &ChatCompletionDiagnostics,
) -> String {
    let progress = total
        .map(|total| format!(" {count}/{total}"))
        .unwrap_or_else(|| format!(" {count}"));
    let reasoning = diagnostics.reasoning_content.as_deref().unwrap_or("");
    format!(
        "{label}: chat completion request{progress} completed model={} finish={} prompt_tokens={} completion_tokens={} total_tokens={}\n\
         {label}: reasoning output ({} chars)\n{}\n\
         {label}: visible output ({} chars)\n{}",
        diagnostics.model,
        diagnostics.finish_reason,
        diagnostics.prompt_tokens,
        diagnostics.completion_tokens,
        diagnostics.total_tokens,
        reasoning.chars().count(),
        diagnostic_excerpt(reasoning),
        diagnostics.visible_content.chars().count(),
        diagnostic_excerpt(&diagnostics.visible_content),
    )
}

fn diagnostic_excerpt(text: &str) -> String {
    const LIMIT: usize = 4096;
    if text.chars().count() <= LIMIT {
        return text.to_string();
    }
    let head = text.chars().take(LIMIT / 2).collect::<String>();
    let tail = text
        .chars()
        .rev()
        .take(LIMIT / 2)
        .collect::<String>()
        .chars()
        .rev()
        .collect::<String>();
    format!("{head}\n[... truncated ...]\n{tail}")
}

fn empty_benchmark() -> crate::ffi::runtime_bindings::MsBaselineBenchmark {
    crate::ffi::runtime_bindings::MsBaselineBenchmark {
        load_ms: 0.0,
        prompt_eval_ms: 0.0,
        generation_ms: 0.0,
        prompt_eval_tps: 0.0,
        token_gen_tps: 0.0,
        ttft_ms: 0.0,
        vram_peak_mb: 0.0,
        vram_allocated_mb: 0.0,
        prompt_tokens: 0,
        generated_tokens: 0,
        copied_tensor_count: 0,
        converted_tensor_count: 0,
        converted_bytes_before: 0,
        converted_bytes_after: 0,
        requested_target_count: 0,
        verified_target_count: 0,
    }
}

fn apply_stop_strings(content: &mut String, stops: &[String]) -> &'static str {
    let Some((index, _)) = stops
        .iter()
        .map(String::as_str)
        .filter(|stop| !stop.is_empty())
        .filter_map(|stop| content.find(stop).map(|index| (index, stop)))
        .min_by_key(|(index, _)| *index)
    else {
        return "stop";
    };
    content.truncate(index);
    "stop"
}

fn chat_finish_reason_label(reason: ChatFinishReason) -> &'static str {
    match reason {
        ChatFinishReason::Stop => "stop",
        ChatFinishReason::Length => "length",
        ChatFinishReason::Eos => "stop",
    }
}

fn json_response(status: u16, reason: &'static str, body: serde_json::Value) -> HttpResponse {
    HttpResponse {
        status,
        reason,
        body,
    }
}

fn write_response(stream: &mut TcpStream, response: HttpResponse) -> Result<(), String> {
    let body = serde_json::to_string(&response.body).map_err(|e| e.to_string())?;
    let response_text = format!(
        "HTTP/1.1 {} {}\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: Authorization, Content-Type\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        response.status,
        response.reason,
        body.as_bytes().len(),
        body
    );
    stream
        .write_all(response_text.as_bytes())
        .map_err(|e| e.to_string())
}

fn write_sse_headers(stream: &mut TcpStream) -> Result<(), String> {
    stream
        .write_all(
            b"HTTP/1.1 200 OK\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: Authorization, Content-Type\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n",
        )
        .map_err(|e| e.to_string())?;
    stream.flush().map_err(|e| e.to_string())
}

fn write_sse_data(stream: &mut TcpStream, value: &Value) -> Result<(), String> {
    let body = serde_json::to_string(value).map_err(|e| e.to_string())?;
    stream
        .write_all(format!("data: {body}\n\n").as_bytes())
        .map_err(|e| e.to_string())?;
    stream.flush().map_err(|e| e.to_string())
}

fn write_sse_done(stream: &mut TcpStream) -> Result<(), String> {
    stream
        .write_all(b"data: [DONE]\n\n")
        .map_err(|e| e.to_string())?;
    stream.flush().map_err(|e| e.to_string())
}

fn write_chat_completion_stream_delta(
    stream: &mut TcpStream,
    id: &str,
    created: u64,
    model: &str,
    visible_delta: &str,
    reasoning_delta: &str,
) -> Result<(), String> {
    if visible_delta.is_empty() && reasoning_delta.is_empty() {
        return Ok(());
    }
    write_sse_data(
        stream,
        &chat_completion_stream_delta_chunk(id, created, model, visible_delta, reasoning_delta),
    )
}

fn write_chat_completion_stream_finish(
    stream: &mut TcpStream,
    id: &str,
    created: u64,
    model: &str,
    finish_reason: &str,
    benchmark: &crate::ffi::runtime_bindings::MsBaselineBenchmark,
) -> Result<(), String> {
    write_sse_data(
        stream,
        &chat_completion_stream_finish_chunk(id, created, model, finish_reason, benchmark),
    )
}

fn chat_completion_stream_delta_chunk(
    id: &str,
    created: u64,
    model: &str,
    visible_delta: &str,
    reasoning_delta: &str,
) -> Value {
    let mut delta = json!({
        "role": "assistant"
    });
    if !visible_delta.is_empty() {
        delta["content"] = json!(visible_delta);
    }
    if !reasoning_delta.is_empty() {
        delta["reasoning_content"] = json!(reasoning_delta);
    }
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": Value::Null
        }]
    })
}

fn chat_completion_stream_finish_chunk(
    id: &str,
    created: u64,
    model: &str,
    finish_reason: &str,
    benchmark: &crate::ffi::runtime_bindings::MsBaselineBenchmark,
) -> Value {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": finish_reason
        }],
        "usage": {
            "prompt_tokens": benchmark.prompt_tokens,
            "completion_tokens": benchmark.generated_tokens,
            "total_tokens": benchmark.prompt_tokens + benchmark.generated_tokens
        }
    })
}

fn model_id_from_path(path: &str) -> String {
    std::path::Path::new(path)
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("modelinspector-local")
        .to_string()
}

fn make_token() -> String {
    format!("modelinspector-{}-{}", std::process::id(), unix_millis())
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
        .collect()
}

fn quant_type_name(quant_type: &QuantType) -> &'static str {
    match quant_type {
        QuantType::F32 => "F32",
        QuantType::BF16 => "BF16",
        QuantType::F16 => "F16",
        QuantType::Q8_0 => "Q8_0",
        QuantType::Q6_K => "Q6_K",
        QuantType::Q5_K => "Q5_K",
        QuantType::Q5_K_M => "Q5_K_M",
        QuantType::Q5_1 => "Q5_1",
        QuantType::Q5_0 => "Q5_0",
        QuantType::Q4_K => "Q4_K",
        QuantType::Q4_K_M => "Q4_K_M",
        QuantType::Q4_1 => "Q4_1",
        QuantType::Q4_0 => "Q4_0",
        QuantType::Q3_K => "Q3_K",
        QuantType::Q3_K_M => "Q3_K_M",
        QuantType::Q2_K => "Q2_K",
    }
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
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
    use std::net::TcpStream;

    fn state() -> HttpApiState {
        HttpApiState {
            token: "test-token".to_string(),
            model_id: "Qwen3.5-9B-Q4_K_M.gguf".to_string(),
            session: None,
            runtime_totals: Arc::new(Mutex::new(ModelInspectorApiRuntimeTotals::default())),
            benchmark_label: None,
            benchmark_sample_count: None,
            default_enable_thinking: None,
            vision_enabled: true,
            completion_count: AtomicU64::new(0),
            app: None,
        }
    }

    fn request(method: &str, path: &str, token: Option<&str>, body: &str) -> HttpRequest {
        HttpRequest {
            method: method.to_string(),
            path: path.to_string(),
            authorization: token.map(|token| format!("Bearer {token}")),
            body: body.to_string(),
        }
    }

    #[test]
    fn rejects_missing_bearer_token() {
        let body = json!({
            "model": "Qwen3.5-9B-Q4_K_M.gguf",
            "messages": [{"role": "user", "content": "Say hello"}]
        })
        .to_string();
        let response = handle_request(
            &request("POST", "/v1/chat/completions", None, &body),
            &state(),
        );

        assert_eq!(response.status, 401);
        assert_eq!(response.body["error"]["type"], "authentication_error");
    }

    #[test]
    fn serves_models_without_auth_like_llama_cpp() {
        let response = handle_request(&request("GET", "/v1/models", None, ""), &state());

        assert_eq!(response.status, 200);
        assert_eq!(response.body["object"], "list");
        assert_eq!(response.body["data"][0]["id"], "Qwen3.5-9B-Q4_K_M.gguf");
    }

    #[test]
    fn serves_openai_models_shape() {
        let response = handle_request(
            &request("GET", "/v1/models", Some("test-token"), ""),
            &state(),
        );

        assert_eq!(response.status, 200);
        assert_eq!(response.body["object"], "list");
        assert_eq!(response.body["data"][0]["id"], "Qwen3.5-9B-Q4_K_M.gguf");
        assert_eq!(response.body["data"][0]["owned_by"], "modelinspector");
    }

    #[test]
    fn serves_openai_chat_completion_shape() {
        let body = json!({
            "model": "Qwen3.5-9B-Q4_K_M.gguf",
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 8
        })
        .to_string();
        let response = handle_request(
            &request("POST", "/v1/chat/completions", Some("test-token"), &body),
            &state(),
        );

        assert_eq!(response.status, 200);
        assert_eq!(response.body["object"], "chat.completion");
        assert_eq!(response.body["choices"][0]["message"]["role"], "assistant");
        assert!(response.body["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .contains("Model Inspector API smoke response"));
    }

    #[test]
    fn formats_reasoning_content_separately_from_visible_content() {
        let message = assistant_message_json("visible answer", Some("hidden chain"));

        assert_eq!(message["role"], "assistant");
        assert_eq!(message["content"], "visible answer");
        assert_eq!(message["reasoning_content"], "hidden chain");
    }

    #[test]
    fn detects_streaming_chat_completion_requests() {
        let body = json!({
            "model": "Qwen3.5-9B-Q4_K_M.gguf",
            "messages": [{"role": "user", "content": "Say hello"}],
            "stream": true
        })
        .to_string();
        let request = request("POST", "/v1/chat/completions", Some("test-token"), &body);

        assert!(request_wants_streaming_chat_completion(&request));
    }

    #[test]
    fn formats_openai_compatible_streaming_delta_chunk() {
        let chunk = chat_completion_stream_delta_chunk(
            "chatcmpl-test",
            123,
            "model.gguf",
            "visible",
            "reason",
        );

        assert_eq!(chunk["object"], "chat.completion.chunk");
        assert_eq!(chunk["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(chunk["choices"][0]["delta"]["content"], "visible");
        assert_eq!(chunk["choices"][0]["delta"]["reasoning_content"], "reason");
        assert!(chunk["choices"][0]["finish_reason"].is_null());
    }

    #[test]
    fn parses_llama_server_chat_generation_fields() {
        let payload = serde_json::from_value::<ChatCompletionRequest>(json!({
            "model": "Qwen3.5-9B-Q4_K_M.gguf",
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 100,
            "max_completion_tokens": 12,
            "stop": ["</s>", "<|im_end|>"],
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 40,
            "min_p": 0.05,
            "seed": 42,
            "repeat_penalty": 1.1,
            "frequency_penalty": 0.2,
            "presence_penalty": 0.3,
            "chat_template_kwargs": {"enable_thinking": true},
            "reasoning_format": "deepseek",
            "add_generation_prompt": true
        }))
        .unwrap();

        let config = chat_generation_config(&payload, None);

        assert_eq!(config.stop, vec!["</s>", "<|im_end|>"]);
        assert_eq!(config.temperature, Some(0.2));
        assert_eq!(config.top_p, Some(0.95));
        assert_eq!(config.top_k, Some(40));
        assert_eq!(config.min_p, Some(0.05));
        assert_eq!(config.typical_p, None);
        assert_eq!(config.seed, Some(42));
        assert_eq!(config.repeat_last_n, None);
        assert_eq!(config.repeat_penalty, Some(1.1));
        assert_eq!(config.frequency_penalty, Some(0.2));
        assert_eq!(config.presence_penalty, Some(0.3));
        assert_eq!(config.dry_multiplier, None);
        assert_eq!(config.dry_base, None);
        assert_eq!(config.dry_allowed_length, None);
        assert_eq!(config.dry_penalty_last_n, None);
        assert_eq!(
            config.chat_template_kwargs.as_ref().unwrap()["enable_thinking"],
            true
        );
        assert_eq!(config.reasoning_format.as_deref(), Some("deepseek"));
        assert_eq!(config.add_generation_prompt, Some(true));
    }

    #[test]
    fn maps_chat_generation_config_to_native_llama_sampling_params() {
        let payload = serde_json::from_value::<ChatCompletionRequest>(json!({
            "model": "Qwen3.5-9B-Q4_K_M.gguf",
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 100,
            "temperature": 0.0,
            "top_p": 0.75,
            "top_k": 20,
            "min_p": 0.01,
            "typical_p": 0.9,
            "seed": 1234,
            "repeat_last_n": 32,
            "repeat_penalty": 1.15,
            "frequency_penalty": 0.25,
            "presence_penalty": 0.5,
            "dry_multiplier": 0.8,
            "dry_base": 1.5,
            "dry_allowed_length": 3,
            "dry_penalty_last_n": 64,
            "add_generation_prompt": false
        }))
        .unwrap();

        let params = chat_generation_config(&payload, None).to_native_params();

        assert_eq!(params.max_tokens, API_CHAT_UNTIL_CONTEXT_MAX_TOKENS);
        assert_eq!(params.add_generation_prompt, 0);
        assert_eq!(params.seed, 1234);
        assert_eq!(params.top_k, 20);
        assert_eq!(params.repeat_last_n, 32);
        assert_eq!(params.dry_allowed_length, 3);
        assert_eq!(params.dry_penalty_last_n, 64);
        assert_eq!(params.temperature, 0.0);
        assert_eq!(params.top_p, 0.75);
        assert_eq!(params.min_p, 0.01);
        assert_eq!(params.typical_p, 0.9);
        assert_eq!(params.repeat_penalty, 1.15);
        assert_eq!(params.frequency_penalty, 0.25);
        assert_eq!(params.presence_penalty, 0.5);
        assert_eq!(params.dry_multiplier, 0.8);
        assert_eq!(params.dry_base, 1.5);
    }

    #[test]
    fn maps_native_chat_finish_reasons_to_openai_labels() {
        assert_eq!(chat_finish_reason_label(ChatFinishReason::Stop), "stop");
        assert_eq!(chat_finish_reason_label(ChatFinishReason::Length), "length");
        assert_eq!(chat_finish_reason_label(ChatFinishReason::Eos), "stop");
    }

    #[test]
    fn stop_strings_are_forwarded_to_native_generation() {
        let payload = serde_json::from_value::<ChatCompletionRequest>(json!({
            "model": "Qwen3.5-9B-Q4_K_M.gguf",
            "messages": [{"role": "user", "content": "Say hello"}],
            "stop": ["</s>", "<|im_end|>"]
        }))
        .unwrap();

        let config = chat_generation_config(&payload, None);

        assert_eq!(config.native_stop_strings(), vec!["</s>", "<|im_end|>"]);
    }

    #[test]
    fn chat_template_options_are_serialized_for_native_generation() {
        let payload = serde_json::from_value::<ChatCompletionRequest>(json!({
            "model": "Qwen3.5-9B-Q4_K_M.gguf",
            "messages": [{"role": "user", "content": "Say hello"}],
            "chat_template_kwargs": {"enable_thinking": false},
            "reasoning_format": "deepseek"
        }))
        .unwrap();

        let config = chat_generation_config(&payload, None);

        assert_eq!(
            config.native_chat_template_kwargs_json().as_deref(),
            Some("{\"enable_thinking\":false}")
        );
        assert_eq!(config.native_reasoning_format(), Some("deepseek"));
    }

    #[test]
    fn defaults_chat_template_thinking_to_disabled_when_request_omits_it() {
        let payload = serde_json::from_value::<ChatCompletionRequest>(json!({
            "model": "Qwen3.5-9B-Q4_K_M.gguf",
            "messages": [{"role": "user", "content": "Say hello"}]
        }))
        .unwrap();

        let config = chat_generation_config(&payload, None);
        let kwargs: Value =
            serde_json::from_str(config.native_chat_template_kwargs_json().unwrap().as_str())
                .unwrap();

        assert_eq!(kwargs["enable_thinking"], false);
        assert_eq!(config.thinking_override, None);
    }

    #[test]
    fn applies_configured_template_thinking_default_when_request_omits_it() {
        let payload = serde_json::from_value::<ChatCompletionRequest>(json!({
            "model": "Qwen3.5-9B-Q4_K_M.gguf",
            "messages": [{"role": "user", "content": "Say hello"}],
            "chat_template_kwargs": {"foo": "bar"}
        }))
        .unwrap();

        let config = chat_generation_config(&payload, Some(true));
        let kwargs: Value =
            serde_json::from_str(config.native_chat_template_kwargs_json().unwrap().as_str())
                .unwrap();

        assert_eq!(kwargs["foo"], "bar");
        assert_eq!(kwargs["enable_thinking"], true);
        assert_eq!(config.thinking_override, None);
    }

    #[test]
    fn preserves_request_template_thinking_and_records_override() {
        let payload = serde_json::from_value::<ChatCompletionRequest>(json!({
            "model": "Qwen3.5-9B-Q4_K_M.gguf",
            "messages": [{"role": "user", "content": "Say hello"}],
            "chat_template_kwargs": {"enable_thinking": false}
        }))
        .unwrap();

        let config = chat_generation_config(&payload, Some(true));
        let kwargs: Value =
            serde_json::from_str(config.native_chat_template_kwargs_json().unwrap().as_str())
                .unwrap();

        assert_eq!(kwargs["enable_thinking"], false);
        assert_eq!(
            config.thinking_override,
            Some(ChatTemplateThinkingOverride {
                configured: true,
                requested: false,
            })
        );
    }

    #[test]
    fn accepts_openai_text_content_parts_for_chat_messages() {
        let payload = serde_json::from_value::<ChatCompletionRequest>(json!({
            "model": "Qwen3.5-9B-Q4_K_M.gguf",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Say "},
                    {"type": "text", "text": "hello"}
                ]
            }]
        }))
        .unwrap();

        let messages = chat_messages_for_generation(payload.messages.as_deref()).unwrap();

        assert_eq!(
            messages.messages,
            vec![("user".to_string(), "Say hello".to_string())]
        );
        assert!(messages.image_data_urls.is_empty());
    }

    #[test]
    fn accepts_openai_data_image_content_parts_for_chat_messages() {
        let payload = serde_json::from_value::<ChatCompletionRequest>(json!({
            "model": "Qwen3.5-9B-Q4_K_M.gguf",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}
                ]
            }]
        }))
        .unwrap();

        let messages = chat_messages_for_generation(payload.messages.as_deref()).unwrap();

        assert_eq!(
            messages.messages,
            vec![(
                "user".to_string(),
                format!("What is in this image?{MULTIMODAL_IMAGE_MARKER}")
            )]
        );
        assert_eq!(messages.image_data_urls.len(), 1);
        assert!(messages.image_data_urls[0].starts_with("data:image/png;base64,"));
    }

    #[test]
    fn rejects_image_input_without_a_loaded_projector() {
        let body = json!({
            "model": "Qwen3.5-9B-Q4_K_M.gguf",
            "messages": [{
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}]
            }]
        })
        .to_string();
        let mut no_projector_state = state();
        no_projector_state.vision_enabled = false;

        let response = handle_request(
            &request("POST", "/v1/chat/completions", Some("test-token"), &body),
            &no_projector_state,
        );

        assert_eq!(response.status, 400);
        assert!(response.body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("MMPROJ"));
    }

    #[test]
    fn rejects_remote_image_urls_for_chat_messages() {
        let payload = serde_json::from_value::<ChatCompletionRequest>(json!({
            "messages": [{
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}]
            }]
        }))
        .unwrap();

        let error = chat_messages_for_generation(payload.messages.as_deref()).unwrap_err();

        assert!(error.contains("remote URLs are not supported"));
    }

    #[test]
    fn smoke_serves_models_over_loopback_tcp() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let server_state = Arc::new(state());
        let thread_stop = stop.clone();
        let thread_state = server_state.clone();
        let handle = thread::spawn(move || run_server(listener, thread_stop, thread_state));

        let mut stream = TcpStream::connect(addr).unwrap();
        let request = concat!(
            "GET /v1/models HTTP/1.1\r\n",
            "Host: 127.0.0.1\r\n",
            "Authorization: Bearer test-token\r\n",
            "Connection: close\r\n",
            "\r\n"
        );
        stream.write_all(request.as_bytes()).unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();

        stop.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect(addr);
        handle.join().unwrap();

        assert!(response.starts_with("HTTP/1.1 200 OK"));
        assert!(response.contains("\"object\":\"list\""));
        assert!(response.contains("\"owned_by\":\"modelinspector\""));
    }

    #[test]
    fn smoke_streams_chat_completion_over_loopback_tcp() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let server_state = Arc::new(state());
        let thread_stop = stop.clone();
        let thread_state = server_state.clone();
        let handle = thread::spawn(move || run_server(listener, thread_stop, thread_state));
        let body = json!({
            "model": "Qwen3.5-9B-Q4_K_M.gguf",
            "messages": [{"role": "user", "content": "Say hello"}],
            "stream": true
        })
        .to_string();

        let mut stream = TcpStream::connect(addr).unwrap();
        let request = format!(
            "POST /v1/chat/completions HTTP/1.1\r\n\
             Host: 127.0.0.1\r\n\
             Authorization: Bearer test-token\r\n\
             Content-Type: application/json\r\n\
             Content-Length: {}\r\n\
             Connection: close\r\n\
             \r\n\
             {}",
            body.as_bytes().len(),
            body
        );
        stream.write_all(request.as_bytes()).unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();

        stop.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect(addr);
        handle.join().unwrap();

        assert!(response.starts_with("HTTP/1.1 200 OK"));
        assert!(response.contains("Content-Type: text/event-stream"));
        assert!(response.contains("data: {"));
        assert!(response.contains("\"object\":\"chat.completion.chunk\""));
        assert!(response.contains("\"content\":"));
        assert!(response.contains("\"finish_reason\":\"stop\""));
        assert!(response.contains("data: [DONE]"));
    }

    #[test]
    fn formats_benchmark_completion_output_as_api_diagnostic_only() {
        let diagnostics = ChatCompletionDiagnostics {
            model: "gemma".to_string(),
            finish_reason: "stop",
            prompt_tokens: 12,
            completion_tokens: 34,
            total_tokens: 46,
            visible_content: "ANSWER: C".to_string(),
            reasoning_content: Some("private reasoning".to_string()),
        };

        let output = benchmark_completion_output("ModelInspector API", 12, Some(198), &diagnostics);

        assert!(output.contains("ModelInspector API: chat completion request 12/198 completed"));
        assert!(output.contains("finish=stop"));
        assert!(output.contains("visible output"));
        assert!(output.contains("ANSWER: C"));
        assert!(output.contains("reasoning output"));
        assert!(output.contains("private reasoning"));
        assert!(output.find("reasoning output").unwrap() < output.find("visible output").unwrap());
    }

    #[test]
    fn formats_streaming_delta_output_with_reasoning_before_visible() {
        let output =
            streaming_completion_delta_outputs("ModelInspector API", 3, Some(10), "ANSWER", "why");

        assert_eq!(
            output,
            vec![
                "ModelInspector API: chat completion request 3/10 reasoning delta\nwhy",
                "ModelInspector API: chat completion request 3/10 visible delta\nANSWER",
            ]
        );
    }

    #[test]
    fn api_startup_lifecycle_blocks_reentry_until_cancelled_start_finishes() {
        let mut lifecycle = ModelInspectorApiLifecycle::default();
        let (first_start, old_server) = lifecycle.begin_start().unwrap();

        assert!(old_server.is_none());
        assert!(matches!(
            lifecycle.begin_start(),
            Err(message) if message.contains("already starting")
        ));

        let old_server = lifecycle.cancel();

        assert!(old_server.is_none());
        assert!(first_start.cancel_requested());
        assert!(matches!(
            lifecycle.begin_start(),
            Err(message) if message.contains("already starting")
        ));
        assert!(!lifecycle.finish_start(&first_start, None));
        assert!(lifecycle.begin_start().is_ok());
    }
}
