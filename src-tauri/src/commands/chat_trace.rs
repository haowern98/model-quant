use std::fs;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

const TRACE_MAGIC: &[u8; 8] = b"MITRACE1";
const TRACE_VERSION: u32 = 2;
const PAGED_TRACE_VERSION: u32 = 3;

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
    #[serde(default)]
    pub probability: Option<f32>,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatTraceTokenSummary {
    pub index: u32,
    pub token_id: i32,
    pub token_text: String,
    pub logit: f32,
    pub rank: u32,
    pub logit_normalizer: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatTraceManifest {
    pub supported: bool,
    pub tokens: Vec<ChatTraceTokenSummary>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TraceHeader {
    version: u32,
    conversation_id: String,
    assistant_message_id: String,
    model_fingerprint: String,
    #[serde(default)]
    supported: Option<bool>,
    #[serde(default)]
    tokens: Vec<ChatTraceTokenSummary>,
    #[serde(default)]
    token_page_lengths: Vec<u64>,
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

pub fn save_paged_trace_artifact(
    conversation_id: &str,
    assistant_message_id: &str,
    model_fingerprint: &str,
    supported: bool,
    tokens: Vec<ChatTraceToken>,
) -> Result<PathBuf, String> {
    save_paged_trace_artifact_in(
        &trace_directory(),
        conversation_id,
        assistant_message_id,
        model_fingerprint,
        supported,
        tokens,
    )
}

pub fn encode_trace_payload(payload: &ChatTracePayload) -> Result<Vec<u8>, String> {
    serde_json::to_vec(payload).map_err(|error| error.to_string())
}

pub fn decode_trace_payload(payload: &[u8]) -> Result<ChatTracePayload, String> {
    serde_json::from_slice(payload).map_err(|_| "Trace payload is not valid.".to_string())
}

#[tauri::command]
pub fn load_chat_trace_manifest(
    conversation_id: String,
    assistant_message_id: String,
) -> Result<ChatTraceManifest, String> {
    load_trace_manifest_from(&trace_directory(), &conversation_id, &assistant_message_id)
}

#[tauri::command]
pub fn load_chat_trace_token(
    conversation_id: String,
    assistant_message_id: String,
    token_index: usize,
) -> Result<ChatTraceToken, String> {
    load_trace_token_page_from(
        &trace_directory(),
        &conversation_id,
        &assistant_message_id,
        token_index,
    )
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
        supported: None,
        tokens: Vec::new(),
        token_page_lengths: Vec::new(),
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

fn save_paged_trace_artifact_in(
    directory: &Path,
    conversation_id: &str,
    assistant_message_id: &str,
    model_fingerprint: &str,
    supported: bool,
    tokens: Vec<ChatTraceToken>,
) -> Result<PathBuf, String> {
    if !valid_id(conversation_id) || !valid_id(assistant_message_id) || model_fingerprint.trim().is_empty() {
        return Err("Trace artifact is invalid.".to_string());
    }
    let token_page_lengths = tokens
        .iter()
        .map(|token| {
            let mut counter = ByteCounter::default();
            serde_json::to_writer(&mut counter, token).map_err(|error| error.to_string())?;
            Ok(counter.0)
        })
        .collect::<Result<Vec<_>, String>>()?;
    let header = TraceHeader {
        version: PAGED_TRACE_VERSION,
        conversation_id: conversation_id.to_string(),
        assistant_message_id: assistant_message_id.to_string(),
        model_fingerprint: model_fingerprint.to_string(),
        supported: Some(supported),
        tokens: tokens
            .iter()
            .map(|token| ChatTraceTokenSummary {
                index: token.index,
                token_id: token.token_id,
                token_text: token.token_text.clone(),
                logit: token.logit,
                rank: token.rank,
                logit_normalizer: token.logit_normalizer,
            })
            .collect(),
        token_page_lengths,
    };
    let header = serde_json::to_vec(&header).map_err(|error| error.to_string())?;
    let header_len = u32::try_from(header.len()).map_err(|_| "Trace header is too large.".to_string())?;
    let path = trace_artifact_path(directory, conversation_id, assistant_message_id)?;
    let parent = path.parent().ok_or("Trace path is invalid.")?;
    fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    let temporary = path.with_extension("trace.tmp");
    let mut file = BufWriter::new(fs::File::create(&temporary).map_err(|error| error.to_string())?);
    file.write_all(TRACE_MAGIC).map_err(|error| error.to_string())?;
    file.write_all(&header_len.to_le_bytes()).map_err(|error| error.to_string())?;
    file.write_all(&header).map_err(|error| error.to_string())?;
    for token in tokens {
        serde_json::to_writer(&mut file, &token).map_err(|error| error.to_string())?;
    }
    file.flush().map_err(|error| error.to_string())?;
    fs::rename(&temporary, &path).map_err(|error| error.to_string())?;
    Ok(path)
}

#[derive(Default)]
struct ByteCounter(u64);

impl Write for ByteCounter {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        self.0 += buffer.len() as u64;
        Ok(buffer.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn load_trace_manifest_from(
    directory: &Path,
    conversation_id: &str,
    assistant_message_id: &str,
) -> Result<ChatTraceManifest, String> {
    let (header, payload_start) = read_trace_header(directory, conversation_id, assistant_message_id)?;
    if header.version == PAGED_TRACE_VERSION {
        return Ok(ChatTraceManifest {
            supported: header.supported.unwrap_or(false),
            tokens: header.tokens,
        });
    }
    let payload = read_trace_payload(directory, conversation_id, assistant_message_id, payload_start)?;
    let legacy = decode_trace_payload(&payload)?;
    Ok(ChatTraceManifest {
        supported: legacy.supported,
        tokens: legacy.tokens.iter().map(token_summary).collect(),
    })
}

fn load_trace_token_page_from(
    directory: &Path,
    conversation_id: &str,
    assistant_message_id: &str,
    token_index: usize,
) -> Result<ChatTraceToken, String> {
    let (header, payload_start) = read_trace_header(directory, conversation_id, assistant_message_id)?;
    if header.version == PAGED_TRACE_VERSION {
        let length = *header.token_page_lengths.get(token_index).ok_or("Trace token index is invalid.")?;
        let offset = header.token_page_lengths.iter().take(token_index).try_fold(payload_start as u64, |offset, length| {
            offset.checked_add(*length).ok_or("Trace artifact is not valid.")
        })?;
        let path = trace_artifact_path(directory, conversation_id, assistant_message_id)?;
        let mut file = fs::File::open(path).map_err(|error| error.to_string())?;
        file.seek(SeekFrom::Start(offset)).map_err(|error| error.to_string())?;
        let mut page = vec![0; usize::try_from(length).map_err(|_| "Trace page is too large.")?];
        file.read_exact(&mut page).map_err(|error| error.to_string())?;
        return serde_json::from_slice(&page).map_err(|_| "Trace token page is not valid.".to_string());
    }
    let payload = read_trace_payload(directory, conversation_id, assistant_message_id, payload_start)?;
    decode_trace_payload(&payload)?
        .tokens
        .into_iter()
        .nth(token_index)
        .ok_or("Trace token index is invalid.".to_string())
}

fn token_summary(token: &ChatTraceToken) -> ChatTraceTokenSummary {
    ChatTraceTokenSummary {
        index: token.index,
        token_id: token.token_id,
        token_text: token.token_text.clone(),
        logit: token.logit,
        rank: token.rank,
        logit_normalizer: token.logit_normalizer,
    }
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

fn read_trace_header(
    directory: &Path,
    conversation_id: &str,
    assistant_message_id: &str,
) -> Result<(TraceHeader, usize), String> {
    let path = trace_artifact_path(directory, conversation_id, assistant_message_id)?;
    let mut file = fs::File::open(path).map_err(|error| error.to_string())?;
    let mut prefix = [0; TRACE_MAGIC.len() + 4];
    file.read_exact(&mut prefix).map_err(|error| error.to_string())?;
    if &prefix[..TRACE_MAGIC.len()] != TRACE_MAGIC {
        return Err("Trace artifact is not valid.".to_string());
    }
    let header_len = u32::from_le_bytes(prefix[TRACE_MAGIC.len()..].try_into().unwrap()) as usize;
    let mut header = vec![0; header_len];
    file.read_exact(&mut header).map_err(|error| error.to_string())?;
    let header: TraceHeader = serde_json::from_slice(&header).map_err(|_| "Trace artifact is not valid.".to_string())?;
    if !matches!(header.version, TRACE_VERSION | PAGED_TRACE_VERSION)
        || header.conversation_id != conversation_id
        || header.assistant_message_id != assistant_message_id
    {
        return Err("Trace artifact is not valid.".to_string());
    }
    if header.version == PAGED_TRACE_VERSION && header.tokens.len() != header.token_page_lengths.len() {
        return Err("Trace artifact is not valid.".to_string());
    }
    Ok((header, TRACE_MAGIC.len() + 4 + header_len))
}

fn read_trace_payload(
    directory: &Path,
    conversation_id: &str,
    assistant_message_id: &str,
    payload_start: usize,
) -> Result<Vec<u8>, String> {
    let bytes = fs::read(trace_artifact_path(directory, conversation_id, assistant_message_id)?)
        .map_err(|error| error.to_string())?;
    bytes.get(payload_start..)
        .map(ToOwned::to_owned)
        .ok_or("Trace artifact is not valid.".to_string())
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
        load_trace_manifest_from, load_trace_token_page_from, save_paged_trace_artifact_in,
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
                    probability: Some(0.61),
                    token_text: "hello".to_string(),
                }],
                layers: vec![ChatTraceLayer {
                    layer: 0,
                    candidates: vec![ChatTraceCandidate {
                        token_id: 42,
                        logit: 3.0,
                        probability: Some(0.61),
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

    #[test]
    fn reads_a_compact_manifest_and_only_the_requested_token_page() {
        let directory =
            std::env::temp_dir().join(format!("model-surgery-paged-trace-test-{}", std::process::id()));
        let token = |index, token_id, token_text: &str| ChatTraceToken {
            index,
            token_id,
            token_text: token_text.to_string(),
            logit: index as f32,
            rank: index,
            logit_normalizer: index as f64,
            candidates: vec![ChatTraceCandidate {
                token_id,
                logit: index as f32,
                probability: Some(0.5),
                token_text: token_text.to_string(),
            }],
            layers: vec![ChatTraceLayer {
                layer: index as i32,
                candidates: vec![ChatTraceCandidate {
                    token_id,
                    logit: index as f32,
                    probability: Some(0.5),
                    token_text: token_text.to_string(),
                }],
            }],
        };

        save_paged_trace_artifact_in(
            &directory,
            "conversation-1",
            "assistant-1",
            "model-fingerprint",
            true,
            vec![token(1, 41, "first"), token(2, 42, "second")],
        )
        .unwrap();

        let manifest = load_trace_manifest_from(&directory, "conversation-1", "assistant-1").unwrap();
        assert!(manifest.supported);
        assert_eq!(manifest.tokens.len(), 2);
        assert_eq!(manifest.tokens[0].token_text, "first");
        assert_eq!(manifest.tokens[1].token_id, 42);

        let page = load_trace_token_page_from(&directory, "conversation-1", "assistant-1", 1).unwrap();
        assert_eq!(page.layers.len(), 1);
        assert_eq!(page.layers[0].layer, 2);
        assert_eq!(page.layers[0].candidates[0].token_text, "second");

        fs::remove_dir_all(directory).unwrap();
    }
}
