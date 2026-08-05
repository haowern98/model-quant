use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

const TRACE_MAGIC: &[u8; 8] = b"MITRACE1";
const TRACE_VERSION: u32 = 2;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatTraceArtifact {
    pub conversation_id: String,
    pub assistant_message_id: String,
    pub model_fingerprint: String,
    pub payload: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatTraceCandidate {
    pub token_id: i32,
    pub logit: f32,
    pub token_text: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatTraceLayer {
    pub layer: i32,
    pub candidates: Vec<ChatTraceCandidate>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatTraceToken {
    pub index: u32,
    pub token_id: i32,
    pub token_text: String,
    pub logit: f32,
    pub rank: u32,
    pub logit_normalizer: f64,
    pub candidates: Vec<ChatTraceCandidate>,
    pub layers: Vec<ChatTraceLayer>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatTracePayload {
    pub supported: bool,
    pub tokens: Vec<ChatTraceToken>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TraceHeader {
    version: u32,
    conversation_id: String,
    assistant_message_id: String,
    model_fingerprint: String,
}

pub fn save_trace_artifact(artifact: &ChatTraceArtifact) -> Result<PathBuf, String> {
    save_trace_artifact_in(&trace_directory(), artifact)
}

pub fn load_trace_artifact(
    conversation_id: &str,
    assistant_message_id: &str,
) -> Result<ChatTraceArtifact, String> {
    load_trace_artifact_from(&trace_directory(), conversation_id, assistant_message_id)
}

pub fn encode_trace_payload(payload: &ChatTracePayload) -> Result<Vec<u8>, String> {
    serde_json::to_vec(payload).map_err(|error| error.to_string())
}

pub fn decode_trace_payload(payload: &[u8]) -> Result<ChatTracePayload, String> {
    serde_json::from_slice(payload).map_err(|_| "Trace payload is not valid.".to_string())
}

#[tauri::command]
pub fn load_chat_trace(
    conversation_id: String,
    assistant_message_id: String,
) -> Result<ChatTracePayload, String> {
    let artifact = load_trace_artifact(&conversation_id, &assistant_message_id)?;
    decode_trace_payload(&artifact.payload)
}

fn save_trace_artifact_in(
    directory: &Path,
    artifact: &ChatTraceArtifact,
) -> Result<PathBuf, String> {
    validate_artifact(artifact)?;
    let header = TraceHeader {
        version: TRACE_VERSION,
        conversation_id: artifact.conversation_id.clone(),
        assistant_message_id: artifact.assistant_message_id.clone(),
        model_fingerprint: artifact.model_fingerprint.clone(),
    };
    let header = serde_json::to_vec(&header).map_err(|error| error.to_string())?;
    let header_len =
        u32::try_from(header.len()).map_err(|_| "Trace header is too large.".to_string())?;
    let mut bytes =
        Vec::with_capacity(TRACE_MAGIC.len() + 4 + header.len() + artifact.payload.len());
    bytes.extend_from_slice(TRACE_MAGIC);
    bytes.extend_from_slice(&header_len.to_le_bytes());
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&artifact.payload);

    let path = trace_artifact_path(
        directory,
        &artifact.conversation_id,
        &artifact.assistant_message_id,
    )?;
    let parent = path.parent().ok_or("Trace path is invalid.")?;
    fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    fs::write(&path, bytes).map_err(|error| error.to_string())?;
    Ok(path)
}

fn load_trace_artifact_from(
    directory: &Path,
    conversation_id: &str,
    assistant_message_id: &str,
) -> Result<ChatTraceArtifact, String> {
    let bytes = fs::read(trace_artifact_path(
        directory,
        conversation_id,
        assistant_message_id,
    )?)
    .map_err(|error| error.to_string())?;
    if bytes.len() < TRACE_MAGIC.len() + 4 || &bytes[..TRACE_MAGIC.len()] != TRACE_MAGIC {
        return Err("Trace artifact is not valid.".to_string());
    }
    let header_start = TRACE_MAGIC.len() + 4;
    let header_len =
        u32::from_le_bytes(bytes[TRACE_MAGIC.len()..header_start].try_into().unwrap()) as usize;
    let header_end = header_start
        .checked_add(header_len)
        .ok_or("Trace artifact is not valid.")?;
    let header = bytes
        .get(header_start..header_end)
        .ok_or("Trace artifact is not valid.")?;
    let header: TraceHeader =
        serde_json::from_slice(header).map_err(|_| "Trace artifact is not valid.".to_string())?;
    if header.version != TRACE_VERSION
        || header.conversation_id != conversation_id
        || header.assistant_message_id != assistant_message_id
    {
        return Err("Trace artifact is not valid.".to_string());
    }
    Ok(ChatTraceArtifact {
        conversation_id: header.conversation_id,
        assistant_message_id: header.assistant_message_id,
        model_fingerprint: header.model_fingerprint,
        payload: bytes[header_end..].to_vec(),
    })
}

fn trace_directory() -> PathBuf {
    trace_directory_from_local_app_data(std::env::var_os("LOCALAPPDATA"))
}

fn trace_directory_from_local_app_data(local_app_data: Option<std::ffi::OsString>) -> PathBuf {
    local_app_data
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join("MI")
        .join("g")
        .join("traces")
}

fn trace_artifact_path(
    directory: &Path,
    conversation_id: &str,
    assistant_message_id: &str,
) -> Result<PathBuf, String> {
    if !valid_id(conversation_id) || !valid_id(assistant_message_id) {
        return Err("Trace artifact id is invalid.".to_string());
    }
    Ok(directory
        .join(conversation_id)
        .join(format!("{assistant_message_id}.trace")))
}

fn validate_artifact(artifact: &ChatTraceArtifact) -> Result<(), String> {
    if !valid_id(&artifact.conversation_id)
        || !valid_id(&artifact.assistant_message_id)
        || artifact.model_fingerprint.trim().is_empty()
    {
        return Err("Trace artifact is invalid.".to_string());
    }
    Ok(())
}

fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 128
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
}

#[cfg(test)]
mod tests {
    use super::{
        decode_trace_payload, encode_trace_payload, load_trace_artifact_from,
        save_trace_artifact_in, trace_directory_from_local_app_data, ChatTraceArtifact,
        ChatTraceCandidate, ChatTraceLayer, ChatTracePayload, ChatTraceToken,
    };
    use std::ffi::OsString;
    use std::fs;
    use std::path::PathBuf;

    #[test]
    fn stores_trace_artifacts_in_the_mi_local_app_data_directory() {
        assert_eq!(
            trace_directory_from_local_app_data(Some(OsString::from(
                r"C:\\Users\\tester\\AppData\\Local"
            ))),
            PathBuf::from(r"C:\\Users\\tester\\AppData\\Local")
                .join("MI")
                .join("g")
                .join("traces"),
        );
    }

    #[test]
    fn round_trips_a_trace_artifact() {
        let directory =
            std::env::temp_dir().join(format!("model-surgery-trace-test-{}", std::process::id()));
        let artifact = ChatTraceArtifact {
            conversation_id: "conversation-1".to_string(),
            assistant_message_id: "assistant-1".to_string(),
            model_fingerprint: "model-fingerprint".to_string(),
            payload: vec![1, 2, 3],
        };

        save_trace_artifact_in(&directory, &artifact).unwrap();
        assert_eq!(
            load_trace_artifact_from(&directory, "conversation-1", "assistant-1").unwrap(),
            artifact
        );
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn round_trips_generated_token_trace_data() {
        let payload = ChatTracePayload {
            supported: true,
            tokens: vec![ChatTraceToken {
                index: 1,
                token_id: 42,
                token_text: "hello".to_string(),
                logit: 3.0,
                rank: 2,
                logit_normalizer: 3.5,
                candidates: vec![ChatTraceCandidate {
                    token_id: 42,
                    logit: 3.0,
                    token_text: "hello".to_string(),
                }],
                layers: vec![ChatTraceLayer {
                    layer: 0,
                    candidates: vec![ChatTraceCandidate {
                        token_id: 42,
                        logit: 3.0,
                        token_text: "hello".to_string(),
                    }],
                }],
            }],
        };

        assert_eq!(
            decode_trace_payload(&encode_trace_payload(&payload).unwrap()).unwrap(),
            payload
        );
    }
}
