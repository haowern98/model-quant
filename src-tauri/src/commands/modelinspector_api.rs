use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc, Mutex,
};
use std::thread::{self, JoinHandle};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tauri::{AppHandle, State};

use crate::commands::quant::RecipeStore;
use crate::ffi::runtime_bindings::{ChatFinishReason, ChatGenerationParams, RecipeChatSession};
use crate::quant::recipe::{QuantType, RecipeState};

const API_CHAT_MAX_TOKENS: u32 = 10000;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInspectorApiStatus {
    pub running: bool,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub model_id: Option<String>,
}

pub struct ModelInspectorApiState(pub Mutex<Option<ModelInspectorApiServer>>);

impl ModelInspectorApiState {
    pub fn new() -> Self {
        Self(Mutex::new(None))
    }
}

pub struct ModelInspectorApiServer {
    base_url: String,
    model_id: String,
    token: String,
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

#[tauri::command]
pub async fn start_modelinspector_api(
    benchmark_label: Option<String>,
    _benchmark_sample_count: Option<u64>,
    app: AppHandle,
    api_state: State<'_, ModelInspectorApiState>,
    recipe_state: State<'_, RecipeStore>,
) -> Result<ModelInspectorApiStatus, String> {
    let recipe = recipe_state
        .0
        .lock()
        .map_err(|e| e.to_string())?
        .clone()
        .ok_or("No model loaded")?;

    let model_id = model_id_from_path(&recipe.base_model);
    let token = make_token();
    let listener = TcpListener::bind("127.0.0.1:0")
        .map_err(|e| format!("Failed to bind Model Inspector API: {e}"))?;
    let addr = listener
        .local_addr()
        .map_err(|e| format!("Failed to read Model Inspector API address: {e}"))?;
    let base_url = format!("http://{addr}/v1");
    let stop = Arc::new(AtomicBool::new(false));
    let targets = recipe_targets(&recipe);
    crate::progress::emit_benchmark_output(
        &app,
        "ModelInspector API: loading in-process model session",
    );
    let output_app = app.clone();
    let session = crate::ffi::runtime_bindings::open_recipe_chat_session_with_progress(
        &recipe.base_model,
        &targets,
        API_CHAT_MAX_TOKENS,
        |message| {
            crate::progress::emit_benchmark_output(&output_app, message);
        },
    )
    .map_err(|e| format!("Failed to load Model Inspector API model session: {e}"))?;
    let server_state = Arc::new(HttpApiState {
        token: token.clone(),
        model_id: model_id.clone(),
        session: Some(Mutex::new(session)),
        benchmark_label,
        completion_count: AtomicU64::new(0),
        app: Some(app),
    });

    let thread_stop = stop.clone();
    let thread_state = server_state.clone();
    let handle = thread::Builder::new()
        .name("modelinspector-api".to_string())
        .spawn(move || run_server(listener, thread_stop, thread_state))
        .map_err(|e| format!("Failed to start Model Inspector API thread: {e}"))?;

    let mut guard = api_state.0.lock().map_err(|e| e.to_string())?;
    if let Some(mut server) = guard.take() {
        server.stop();
    }
    *guard = Some(ModelInspectorApiServer {
        base_url: base_url.clone(),
        model_id: model_id.clone(),
        token,
        stop,
        handle: Some(handle),
    });
    if let Some(app) = server_state.app.as_ref() {
        crate::progress::emit_benchmark_output(
            app,
            format!("ModelInspector API ready at {base_url}"),
        );
    }

    Ok(ModelInspectorApiStatus {
        running: true,
        base_url: Some(base_url),
        api_key: guard.as_ref().map(|server| server.token.clone()),
        model_id: Some(model_id),
    })
}

#[tauri::command]
pub async fn stop_modelinspector_api(
    api_state: State<'_, ModelInspectorApiState>,
) -> Result<ModelInspectorApiStatus, String> {
    let mut guard = api_state.0.lock().map_err(|e| e.to_string())?;
    if let Some(mut server) = guard.take() {
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
    Ok(match guard.as_ref() {
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
    benchmark_label: Option<String>,
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
    max_tokens: u32,
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
    reasoning_format: Option<String>,
}

fn chat_generation_config(payload: &ChatCompletionRequest) -> ChatGenerationConfig {
    let requested_tokens = payload
        .n_predict
        .and_then(|value| u32::try_from(value).ok())
        .or(payload.max_completion_tokens)
        .or(payload.max_tokens)
        .unwrap_or(API_CHAT_MAX_TOKENS);

    ChatGenerationConfig {
        max_tokens: requested_tokens.min(API_CHAT_MAX_TOKENS),
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
        chat_template_kwargs: payload.chat_template_kwargs.clone(),
        reasoning_format: payload.reasoning_format.clone(),
    }
}

impl ChatGenerationConfig {
    fn to_native_params(&self) -> ChatGenerationParams {
        let defaults = ChatGenerationParams::default();
        ChatGenerationParams {
            max_tokens: self.max_tokens,
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
) -> Result<Vec<(String, String)>, String> {
    let messages = messages.unwrap_or_default();
    messages
        .iter()
        .map(|message| {
            let content = chat_message_content_text(&message.content).ok_or_else(|| {
                "messages must use string content or text content parts".to_string()
            })?;
            Ok((message.role.clone(), content))
        })
        .collect()
}

fn chat_message_content_text(content: &Value) -> Option<String> {
    match content {
        Value::String(text) => Some(text.clone()),
        Value::Array(parts) => {
            let mut text = String::new();
            for part in parts {
                let part_type = part.get("type").and_then(Value::as_str);
                if part_type != Some("text") {
                    return None;
                }
                text.push_str(part.get("text").and_then(Value::as_str).unwrap_or_default());
            }
            Some(text)
        }
        _ => None,
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

    let config = chat_generation_config(&payload);
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
    let max_tokens = native_params.max_tokens;
    let model = payload.model.unwrap_or_else(|| state.model_id.clone());
    if messages.is_empty()
        || messages
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

    let (mut content, benchmark, mut finish_reason) = match state.session.as_ref() {
        Some(session) => match session
            .lock()
            .map_err(|e| e.to_string())
            .and_then(|mut session| {
                session.generate_chat(
                    &messages,
                    &native_params,
                    &config.native_stop_strings(),
                    config.native_chat_template_kwargs_json().as_deref(),
                    config.native_reasoning_format(),
                )
            })
        {
            Ok(output) => (
                output.text,
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
        },
        None => {
            let prompt_chars = messages
                .iter()
                .map(|(_, content)| content.chars().count())
                .sum::<usize>();
            (
                format!(
                    "Model Inspector API smoke response for {model}. Received {prompt_chars} prompt character(s), max_tokens={max_tokens}."
                ),
                empty_benchmark(),
                "stop",
            )
        }
    };
    if state.session.is_none() {
        finish_reason = apply_stop_strings(&mut content, &config.stop);
    }
    emit_completion_output(state);

    json_response(
        200,
        "OK",
        json!({
            "id": format!("chatcmpl-{}", unix_millis()),
            "object": "chat.completion",
            "created": unix_seconds(),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": finish_reason
            }],
            "usage": {
                "prompt_tokens": benchmark.prompt_tokens,
                "completion_tokens": benchmark.generated_tokens,
                "total_tokens": benchmark.prompt_tokens + benchmark.generated_tokens
            }
        }),
    )
}

fn emit_completion_output(state: &HttpApiState) {
    let Some(label) = state.benchmark_label.as_deref() else {
        return;
    };
    let count = state.completion_count.fetch_add(1, Ordering::SeqCst) + 1;
    if let Some(app) = state.app.as_ref() {
        crate::progress::emit_benchmark_output(
            app,
            benchmark_completion_output(label, count, None),
        );
    }
}

fn benchmark_completion_output(label: &str, count: u64, total: Option<u64>) -> String {
    let _ = total;
    format!("{label}: chat completion request {count} completed")
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
        "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        response.status,
        response.reason,
        body.as_bytes().len(),
        body
    );
    stream
        .write_all(response_text.as_bytes())
        .map_err(|e| e.to_string())
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
        QuantType::Q4_K => "Q4_K",
        QuantType::Q4_K_M => "Q4_K_M",
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
            benchmark_label: None,
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
    fn rejects_streaming_until_it_is_supported() {
        let body = json!({
            "model": "Qwen3.5-9B-Q4_K_M.gguf",
            "messages": [{"role": "user", "content": "Say hello"}],
            "stream": true
        })
        .to_string();
        let response = handle_request(
            &request("POST", "/v1/chat/completions", Some("test-token"), &body),
            &state(),
        );

        assert_eq!(response.status, 400);
        assert_eq!(response.body["error"]["type"], "invalid_request_error");
        assert!(response.body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("Streaming"));
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

        let config = chat_generation_config(&payload);

        assert_eq!(config.max_tokens, 12);
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

        let params = chat_generation_config(&payload).to_native_params();

        assert_eq!(params.max_tokens, 100);
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

        let config = chat_generation_config(&payload);

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

        let config = chat_generation_config(&payload);

        assert_eq!(
            config.native_chat_template_kwargs_json().as_deref(),
            Some("{\"enable_thinking\":false}")
        );
        assert_eq!(config.native_reasoning_format(), Some("deepseek"));
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
            messages,
            vec![("user".to_string(), "Say hello".to_string())]
        );
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
    fn formats_benchmark_completion_output_as_api_diagnostic_only() {
        assert_eq!(
            benchmark_completion_output("ModelInspector API", 12, Some(198)),
            "ModelInspector API: chat completion request 12 completed"
        );
        assert_eq!(
            benchmark_completion_output("ModelInspector API", 3, None),
            "ModelInspector API: chat completion request 3 completed"
        );
    }
}
