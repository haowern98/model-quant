use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, State};

use crate::commands::quant::RecipeStore;
use crate::ffi::runtime_bindings::{
    open_recipe_chat_session, ChatFinishReason, ChatGenerationParams, RecipeChatSession,
};
use crate::quant::recipe::{QuantType, RecipeState};

const CHAT_STREAM_EVENT: &str = "chat-stream-delta";
const CHAT_KIND: &str = "model-quant-chat";

pub struct ChatRuntimeState(Mutex<Option<ChatRuntime>>);

impl ChatRuntimeState {
    pub fn new() -> Self {
        Self(Mutex::new(None))
    }
}

struct ChatRuntime {
    base_model: String,
    targets: Vec<(String, String)>,
    config: ChatGenerationConfig,
    session: RecipeChatSession,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatGenerationRequest {
    conversation_id: String,
    messages: Vec<ChatGenerationMessage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatGenerationMessage {
    pub role: String,
    pub content: String,
    pub reasoning: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatGenerationConfig {
    seed: Option<u32>,
    thinking: bool,
    temperature: f64,
    top_k: i32,
    repeat_penalty: f64,
    presence_penalty: f64,
    top_p: f64,
    min_p: f64,
    context_window: u32,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatModelLoadStatus {
    pub model: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatGenerationResponse {
    pub model: String,
    pub content: String,
    pub reasoning: Option<String>,
    pub tokens_per_second: f64,
    pub prompt_tokens: u32,
    pub duration_seconds: f64,
    pub finish_reason: String,
    pub seed: u32,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ChatStreamDelta {
    conversation_id: String,
    content: String,
    reasoning: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StoredChatConversation {
    kind: String,
    version: u32,
    id: String,
    title: String,
    created_at: String,
    updated_at: String,
    messages: Vec<StoredChatMessage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StoredChatMessage {
    id: String,
    role: String,
    content: String,
    reasoning: Option<String>,
    model: Option<String>,
    tokens_per_second: Option<f64>,
    prompt_tokens: Option<u32>,
    duration_seconds: Option<f64>,
    finish_reason: Option<String>,
    seed: Option<u32>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatConversationSummary {
    pub id: String,
    pub title: String,
    pub updated_at: String,
}

#[tauri::command]
pub async fn load_chat_model(
    config: ChatGenerationConfig,
    chat_runtime: State<'_, ChatRuntimeState>,
    recipe_state: State<'_, RecipeStore>,
) -> Result<ChatModelLoadStatus, String> {
    validate_config(&config)?;
    let recipe = recipe_state
        .0
        .lock()
        .map_err(|error| error.to_string())?
        .clone()
        .ok_or("Open a GGUF model before loading it for Chat.")?;
    let targets = recipe_targets(&recipe);
    let mut runtime = chat_runtime.0.lock().map_err(|error| error.to_string())?;
    let already_loaded = runtime.as_ref().is_some_and(|current| {
        current.base_model == recipe.base_model && current.targets == targets && current.config == config
    });
    if !already_loaded {
        crate::ffi::runtime_bindings::reset_recipe_test_cancel();
        *runtime = Some(ChatRuntime {
            base_model: recipe.base_model.clone(),
            targets: targets.clone(),
            config: config.clone(),
            session: open_recipe_chat_session(&recipe.base_model, &targets, config.context_window)?,
        });
    }
    Ok(ChatModelLoadStatus { model: model_name(&recipe.base_model) })
}

#[tauri::command]
pub async fn unload_chat_model(
    chat_runtime: State<'_, ChatRuntimeState>,
) -> Result<(), String> {
    *chat_runtime.0.lock().map_err(|error| error.to_string())? = None;
    Ok(())
}

#[tauri::command]
pub async fn generate_chat_response(
    request: ChatGenerationRequest,
    app: AppHandle,
    chat_runtime: State<'_, ChatRuntimeState>,
) -> Result<ChatGenerationResponse, String> {
    validate_generation_request(&request)?;
    generate(&request, Some(&app), &chat_runtime, true, None, false)
}

#[tauri::command]
pub async fn generate_chat_title(
    request: ChatGenerationRequest,
    chat_runtime: State<'_, ChatRuntimeState>,
) -> Result<String, String> {
    validate_generation_request(&request)?;
    let mut title_request = request;
    title_request.messages.push(ChatGenerationMessage {
        role: "user".to_string(),
        content: "Give this conversation a concise title of at most six words. Return only the title."
            .to_string(),
        reasoning: None,
    });
    let output = generate(&title_request, None, &chat_runtime, false, Some(24), true)?;
    Ok(normalise_title(&output.content))
}

#[tauri::command]
pub async fn save_chat_conversation(
    conversation: StoredChatConversation,
) -> Result<ChatConversationSummary, String> {
    validate_conversation(&conversation)?;
    let path = conversation_path(&conversation.id)?;
    fs::create_dir_all(chat_directory()).map_err(|error| error.to_string())?;
    let encoded = serde_json::to_vec_pretty(&conversation).map_err(|error| error.to_string())?;
    fs::write(path, encoded).map_err(|error| error.to_string())?;
    Ok(summary(&conversation))
}

#[tauri::command]
pub async fn list_chat_conversations() -> Result<Vec<ChatConversationSummary>, String> {
    let directory = chat_directory();
    if !directory.exists() {
        return Ok(Vec::new());
    }
    let mut conversations = Vec::new();
    for entry in fs::read_dir(directory).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        if !entry.file_type().map_err(|error| error.to_string())?.is_file() {
            continue;
        }
        let Ok(contents) = fs::read(entry.path()) else {
            continue;
        };
        let Ok(conversation) = serde_json::from_slice::<StoredChatConversation>(&contents) else {
            continue;
        };
        if conversation.kind == CHAT_KIND {
            conversations.push(summary(&conversation));
        }
    }
    conversations.sort_by(|left, right| right.updated_at.cmp(&left.updated_at));
    Ok(conversations)
}

#[tauri::command]
pub async fn load_chat_conversation(id: String) -> Result<StoredChatConversation, String> {
    let contents = fs::read(conversation_path(&id)?).map_err(|error| error.to_string())?;
    let conversation = serde_json::from_slice::<StoredChatConversation>(&contents)
        .map_err(|error| error.to_string())?;
    if conversation.kind != CHAT_KIND || conversation.id != id {
        return Err("Chat conversation file is not valid.".to_string());
    }
    Ok(conversation)
}

fn generate(
    request: &ChatGenerationRequest,
    app: Option<&AppHandle>,
    chat_runtime: &ChatRuntimeState,
    stream: bool,
    max_tokens: Option<u32>,
    title: bool,
) -> Result<ChatGenerationResponse, String> {
    let mut runtime = chat_runtime.0.lock().map_err(|error| error.to_string())?;
    let runtime = runtime
        .as_mut()
        .ok_or("Load a GGUF model before chatting.")?;
    let messages = request
        .messages
        .iter()
        .map(|message| (message.role.clone(), message.content.clone()))
        .collect::<Vec<_>>();
    let config = if title {
        ChatGenerationConfig { seed: None, thinking: false, ..runtime.config.clone() }
    } else {
        runtime.config.clone()
    };
    let params = ChatGenerationParams {
        max_tokens: max_tokens.unwrap_or_default(),
        seed: config.seed.unwrap_or(u32::MAX),
        top_k: config.top_k,
        temperature: config.temperature,
        top_p: config.top_p,
        min_p: config.min_p,
        repeat_penalty: config.repeat_penalty,
        presence_penalty: config.presence_penalty,
        ..ChatGenerationParams::default()
    };
    let template_kwargs = serde_json::json!({ "enable_thinking": config.thinking }).to_string();
    let output = if stream {
        let conversation_id = request.conversation_id.clone();
        runtime.session.generate_chat_streaming(
            &messages,
            &params,
            &[],
            Some(&template_kwargs),
            None,
            |content, reasoning| {
                let app = app.ok_or("Chat streaming requires an app handle.")?;
                app.emit(
                    CHAT_STREAM_EVENT,
                    ChatStreamDelta {
                        conversation_id: conversation_id.clone(),
                        content: content.to_string(),
                        reasoning: reasoning.to_string(),
                    },
                )
                .map_err(|error| error.to_string())
            },
        )?
    } else {
        runtime.session.generate_chat(
            &messages,
            &params,
            &[],
            Some(&template_kwargs),
            None,
        )?
    };
    let duration_seconds = output.benchmark.generation_ms / 1000.0;
    Ok(ChatGenerationResponse {
        model: model_name(&runtime.base_model),
        content: output.text,
        reasoning: output.reasoning_text,
        tokens_per_second: output.benchmark.token_gen_tps,
        prompt_tokens: output.benchmark.prompt_tokens,
        duration_seconds,
        finish_reason: finish_reason(output.finish_reason).to_string(),
        seed: output.actual_seed,
    })
}

fn validate_generation_request(request: &ChatGenerationRequest) -> Result<(), String> {
    if !valid_id(&request.conversation_id) {
        return Err("Chat conversation id is invalid.".to_string());
    }
    if request.messages.is_empty() || request.messages.last().is_none_or(|message| message.content.trim().is_empty()) {
        return Err("Chat message cannot be empty.".to_string());
    }
    if request.messages.iter().any(|message| !matches!(message.role.as_str(), "system" | "user" | "assistant")) {
        return Err("Chat messages use an unsupported role.".to_string());
    }
    Ok(())
}

fn validate_config(config: &ChatGenerationConfig) -> Result<(), String> {
    if config.context_window == 0 {
        return Err("Context window must be greater than 0.".to_string());
    }
    if !(0.0..=2.0).contains(&config.temperature)
        || !(0..=1000).contains(&config.top_k)
        || !(0.0..=3.0).contains(&config.repeat_penalty)
        || !(-2.0..=2.0).contains(&config.presence_penalty)
        || !(0.0..=1.0).contains(&config.top_p)
        || !(0.0..=1.0).contains(&config.min_p)
    {
        return Err("Chat configuration contains an invalid value.".to_string());
    }
    Ok(())
}

fn validate_conversation(conversation: &StoredChatConversation) -> Result<(), String> {
    if conversation.kind != CHAT_KIND || conversation.version != 1 || !valid_id(&conversation.id) {
        return Err("Chat conversation is invalid.".to_string());
    }
    if conversation.title.trim().is_empty() || conversation.messages.is_empty() {
        return Err("Chat conversation is incomplete.".to_string());
    }
    Ok(())
}

fn chat_directory() -> PathBuf {
    std::env::var_os("USERPROFILE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".lmstudio")
        .join("conversations")
}

fn conversation_path(id: &str) -> Result<PathBuf, String> {
    if !valid_id(id) {
        return Err("Chat conversation id is invalid.".to_string());
    }
    Ok(chat_directory().join(format!("model-quant-{id}.conversation.json")))
}

fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 128
        && id.bytes().all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
}

fn summary(conversation: &StoredChatConversation) -> ChatConversationSummary {
    ChatConversationSummary {
        id: conversation.id.clone(),
        title: conversation.title.clone(),
        updated_at: conversation.updated_at.clone(),
    }
}

fn recipe_targets(recipe: &RecipeState) -> Vec<(String, String)> {
    recipe
        .assignments
        .iter()
        .map(|assignment| (assignment.tensor_name.clone(), quant_type_name(&assignment.quant_type).to_string()))
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

fn model_name(path: &str) -> String {
    PathBuf::from(path)
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("local model")
        .to_string()
}

fn finish_reason(reason: ChatFinishReason) -> &'static str {
    match reason {
        ChatFinishReason::Length => "length",
        ChatFinishReason::Stop | ChatFinishReason::Eos => "stop",
    }
}

fn normalise_title(title: &str) -> String {
    title
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim_matches(|character: char| matches!(character, '"' | '\'' | '`' | '.' | ':'))
        .chars()
        .take(80)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::normalise_title;

    #[test]
    fn normalises_generated_chat_titles() {
        assert_eq!(normalise_title("  A  useful\nchat title.  "), "A useful chat title");
    }
}
