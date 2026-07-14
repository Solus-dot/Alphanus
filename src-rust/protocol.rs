use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use uuid::Uuid;

pub const VERSION: u32 = 1;
pub const MAX_FRAME_BYTES: usize = 1024 * 1024;

#[derive(Debug, Clone, Serialize)]
pub struct Request {
    pub protocol_version: u32,
    #[serde(rename = "type")]
    pub kind: String,
    pub request_id: String,
    pub data: Value,
}

impl Request {
    pub fn new(kind: impl Into<String>, data: Value) -> Self {
        Self {
            protocol_version: VERSION,
            kind: kind.into(),
            request_id: Uuid::new_v4().to_string(),
            data,
        }
    }

    pub fn hello() -> Self {
        Self::new(
            "hello",
            json!({"frontend_version":"0.2.0","min_protocol":VERSION,"max_protocol":VERSION}),
        )
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct EventFrame {
    pub protocol_version: u32,
    #[serde(rename = "type")]
    pub kind: String,
    pub request_id: String,
    pub sequence: u64,
    #[serde(default)]
    pub turn_id: Option<String>,
    #[serde(default)]
    pub approval_id: Option<String>,
    #[serde(default)]
    pub data: Value,
}

impl EventFrame {
    pub fn decode(line: &str) -> Result<Self, String> {
        if line.len() > MAX_FRAME_BYTES {
            return Err("runtime frame exceeds 1 MiB".into());
        }
        let frame: Self = serde_json::from_str(line)
            .map_err(|error| format!("invalid runtime frame: {error}"))?;
        if frame.protocol_version != VERSION {
            return Err(format!(
                "unsupported runtime protocol {}; expected {VERSION}",
                frame.protocol_version
            ));
        }
        if frame.kind.is_empty() || frame.request_id.is_empty() {
            return Err("runtime frame is missing type or request_id".into());
        }
        Ok(frame)
    }
}

#[derive(Debug, Clone)]
pub enum BackendEvent {
    Frame(EventFrame),
    Diagnostic(String),
    Exited(Option<i32>),
    ProtocolError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_forward_compatible_event_fields() {
        let frame = EventFrame::decode(
            r#"{"protocol_version":1,"type":"runtime.ready","request_id":"r","sequence":1,"data":{},"future":true}"#,
        )
        .expect("valid frame");
        assert_eq!(frame.kind, "runtime.ready");
        assert_eq!(frame.sequence, 1);
    }

    #[test]
    fn rejects_oversized_frame() {
        let value = "x".repeat(MAX_FRAME_BYTES + 1);
        assert!(EventFrame::decode(&value).is_err());
    }
}
