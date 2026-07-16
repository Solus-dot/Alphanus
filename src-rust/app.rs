use std::collections::VecDeque;
use std::io::{stdout, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use crossterm::cursor::Show;
use crossterm::event::{
    self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
    Event, KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, BeginSynchronizedUpdate, EndSynchronizedUpdate,
    EnterAlternateScreen, LeaveAlternateScreen,
};
use pulldown_cmark::{CodeBlockKind, Event as MarkdownEvent, Parser, Tag, TagEnd};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Position, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{
    Block, Borders, Clear, List, ListItem, ListState, Paragraph, Scrollbar, ScrollbarOrientation,
    ScrollbarState, Wrap,
};
use ratatui::{Frame, Terminal};
use serde_json::{json, Value};
use syntect::easy::HighlightLines;
use syntect::highlighting::ThemeSet;
use syntect::parsing::SyntaxSet;
use syntect::util::LinesWithEndings;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::backend::Backend;
use crate::protocol::{BackendEvent, EventFrame, Request};
use crate::theme::Theme;

const MAX_EVENTS_PER_FRAME: usize = 256;
const MAX_TRANSCRIPT_ITEMS: usize = 4096;
const MAX_STREAM_CHARS: usize = 512 * 1024;
const MAX_ACTIVITY_ITEMS: usize = 256;
const MAX_TOOL_PREVIEW_CHARS: usize = 8_000;
const MAX_TOOL_PREVIEW_LINES: usize = 140;
const MAX_DIAGNOSTICS: usize = 256;
const PASTE_THRESHOLD: usize = 120;
const CANCEL_WINDOW: Duration = Duration::from_secs(2);
const DEFAULT_SIDEBAR_VISIBLE: bool = false;

#[derive(Debug, Clone, Default)]
struct TurnView {
    id: String,
    user: String,
    attachments: String,
    assistant: String,
    activity: Vec<ActivityItem>,
    state: String,
    label: String,
    branch_root: bool,
    parent: String,
}

impl TurnView {
    fn from_value(value: &Value) -> Self {
        let activity = value
            .get("activity")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(ActivityItem::from_value)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        Self {
            id: field(value, "id"),
            user: field(value, "user"),
            attachments: field(value, "attachments"),
            assistant: field(value, "assistant"),
            activity,
            state: field(value, "assistant_state"),
            label: field(value, "label"),
            branch_root: value
                .get("branch_root")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            parent: field(value, "parent"),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct ToolActivity {
    id: String,
    stream_id: String,
    name: String,
    completed: bool,
    failed: bool,
    filepath: String,
    preview: String,
    language: String,
    preview_truncated: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct ActivityItem {
    kind: String,
    text: String,
    tool: ToolActivity,
}

impl ActivityItem {
    fn reasoning(text: String) -> Self {
        Self {
            kind: "reasoning".into(),
            text,
            tool: ToolActivity::default(),
        }
    }

    fn tool(tool: ToolActivity) -> Self {
        Self {
            kind: "tool".into(),
            text: String::new(),
            tool,
        }
    }

    fn from_value(value: &Value) -> Option<Self> {
        match field(value, "kind").as_str() {
            "reasoning" => Some(Self::reasoning(field(value, "text"))),
            "tool" => Some(Self::tool(ToolActivity {
                id: field(value, "id"),
                stream_id: field(value, "stream_id"),
                name: field(value, "name"),
                completed: value
                    .get("completed")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                failed: value
                    .get("failed")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                filepath: field(value, "filepath"),
                preview: field(value, "preview"),
                language: field(value, "language"),
                preview_truncated: value
                    .get("preview_truncated")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
            })),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
struct Approval {
    id: String,
    reason: String,
    target: String,
    command: String,
    cwd: String,
}

#[derive(Debug, Clone)]
struct PasteChunk {
    marker: String,
    text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Focus {
    Transcript,
    Input,
    Tree,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PaletteMode {
    Commands,
    Global,
    Files,
}

#[derive(Debug, Clone)]
enum Popup {
    Palette {
        query: String,
        selected: usize,
        mode: PaletteMode,
    },
    Sessions {
        query: String,
        selected: usize,
        items: Vec<Value>,
    },
    SessionName {
        value: String,
    },
    Theme {
        selected: usize,
        items: Vec<Value>,
    },
    Config {
        value: String,
        cursor: usize,
    },
    Health {
        report: String,
    },
    Code {
        content: String,
    },
    Fatal,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CommandMessage {
    text: String,
    /// Number of conversation turns that existed when this message arrived.
    /// This keeps local command output interleaved with persisted turns.
    turn_position: usize,
}

struct App {
    backend: Backend,
    python: String,
    project_root: Option<String>,
    debug: bool,
    connected: bool,
    should_quit: bool,
    streaming: bool,
    thinking: bool,
    show_details: bool,
    sidebar: bool,
    command_suggestions: bool,
    command_selected: usize,
    focus: Focus,
    input: String,
    cursor: usize,
    paste_chunks: Vec<PasteChunk>,
    transcript: Vec<TurnView>,
    tree: Vec<TurnView>,
    transcript_scroll: u16,
    transcript_max_scroll: u16,
    transcript_auto_follow: bool,
    tree_selected: usize,
    session_title: String,
    session_id: String,
    branch_name: String,
    collaboration_mode: String,
    model_name: String,
    model_state: String,
    model_endpoint: String,
    context_tokens: Option<u64>,
    context_window: Option<u64>,
    status: String,
    command_messages: VecDeque<CommandMessage>,
    diagnostics: VecDeque<String>,
    pending_attachments: Vec<Value>,
    approval: Option<Approval>,
    popup: Option<Popup>,
    theme: Theme,
    themes: Vec<Value>,
    command_catalog: Vec<Value>,
    shortcut_catalog: Vec<Value>,
    last_sequence: u64,
    last_escape: Option<Instant>,
    last_frame: Instant,
    active_turn_id: String,
    clipboard_notice: Option<(String, Instant)>,
    session_delete_armed: Option<String>,
    show_session_manager_on_ready: bool,
    transcript_offset: Option<usize>,
    transcript_previous: Option<usize>,
    tree_offset: Option<usize>,
    tree_previous: Option<usize>,
    transcript_area: Rect,
    sidebar_area: Rect,
    input_area: Rect,
    composer_area: Rect,
    file_button_area: Rect,
    popup_area: Rect,
    popup_list_area: Rect,
    popup_list_offset: usize,
    popup_new_area: Rect,
    approval_area: Rect,
    approval_approve_area: Rect,
    approval_deny_area: Rect,
    command_suggestions_area: Rect,
}

impl App {
    fn new(python: &str, project_root: Option<&str>, debug: bool) -> Result<Self, String> {
        Ok(Self {
            backend: Backend::start(python, project_root, debug)?,
            python: python.into(),
            project_root: project_root.map(str::to_owned),
            debug,
            connected: false,
            should_quit: false,
            streaming: false,
            thinking: true,
            show_details: true,
            sidebar: DEFAULT_SIDEBAR_VISIBLE,
            command_suggestions: false,
            command_selected: 0,
            focus: Focus::Input,
            input: String::new(),
            cursor: 0,
            paste_chunks: Vec::new(),
            transcript: Vec::new(),
            tree: Vec::new(),
            transcript_scroll: 0,
            transcript_max_scroll: 0,
            transcript_auto_follow: true,
            tree_selected: 0,
            session_title: "Starting…".into(),
            session_id: String::new(),
            branch_name: "root".into(),
            collaboration_mode: "execute".into(),
            model_name: String::new(),
            model_state: "connecting".into(),
            model_endpoint: String::new(),
            context_tokens: None,
            context_window: None,
            status: "Connecting to Python runtime…".into(),
            command_messages: VecDeque::new(),
            diagnostics: VecDeque::new(),
            pending_attachments: Vec::new(),
            approval: None,
            popup: None,
            theme: Theme::default(),
            themes: Vec::new(),
            command_catalog: Vec::new(),
            shortcut_catalog: Vec::new(),
            last_sequence: 0,
            last_escape: None,
            last_frame: Instant::now(),
            active_turn_id: String::new(),
            clipboard_notice: None,
            session_delete_armed: None,
            show_session_manager_on_ready: true,
            transcript_offset: None,
            transcript_previous: None,
            tree_offset: None,
            tree_previous: None,
            transcript_area: Rect::default(),
            sidebar_area: Rect::default(),
            input_area: Rect::default(),
            composer_area: Rect::default(),
            file_button_area: Rect::default(),
            popup_area: Rect::default(),
            popup_list_area: Rect::default(),
            popup_list_offset: 0,
            popup_new_area: Rect::default(),
            approval_area: Rect::default(),
            approval_approve_area: Rect::default(),
            approval_deny_area: Rect::default(),
            command_suggestions_area: Rect::default(),
        })
    }

    fn send(&mut self, kind: &str, data: Value) {
        if let Err(error) = self.backend.send(Request::new(kind, data)) {
            self.status = error;
        }
    }

    fn restart_backend(&mut self) {
        self.backend.shutdown();
        match Backend::start(&self.python, self.project_root.as_deref(), self.debug) {
            Ok(backend) => {
                self.backend = backend;
                self.connected = false;
                self.streaming = false;
                self.last_sequence = 0;
                self.popup = None;
                self.show_session_manager_on_ready = false;
                self.status = "Restarting Python runtime…".into();
            }
            Err(error) => {
                self.status = error;
                self.popup = Some(Popup::Fatal);
            }
        }
    }

    fn drain_backend(&mut self) {
        for _ in 0..MAX_EVENTS_PER_FRAME {
            let Ok(event) = self.backend.events.try_recv() else {
                break;
            };
            match event {
                BackendEvent::Frame(frame) => self.apply_frame(frame),
                BackendEvent::Diagnostic(line) => {
                    push_bounded(&mut self.diagnostics, line, MAX_DIAGNOSTICS);
                }
                BackendEvent::ProtocolError(error) => {
                    push_bounded(&mut self.diagnostics, error.clone(), MAX_DIAGNOSTICS);
                    self.status = error;
                }
                BackendEvent::Exited(code) => {
                    if !self.should_quit {
                        self.connected = false;
                        self.streaming = false;
                        self.status = format!(
                            "Python runtime exited ({})",
                            code.map_or_else(|| "signal".into(), |v| v.to_string())
                        );
                        self.popup = Some(Popup::Fatal);
                    }
                }
            }
        }
    }

    fn apply_frame(&mut self, frame: EventFrame) {
        if frame.sequence <= self.last_sequence {
            self.status = format!("Ignored out-of-order runtime event {}", frame.sequence);
            return;
        }
        self.last_sequence = frame.sequence;
        match frame.kind.as_str() {
            "runtime.ready" => {
                self.connected = true;
                self.status = "Ready".into();
                self.command_catalog = frame
                    .data
                    .get("commands")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                self.shortcut_catalog = frame
                    .data
                    .get("shortcuts")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                if let Some(snapshot) = frame.data.get("snapshot") {
                    self.apply_snapshot(snapshot);
                }
                self.send("theme.list", json!({}));
                self.send("palette.get", json!({}));
                self.send("status.refresh", json!({}));
                if take_startup_session_manager(&mut self.show_session_manager_on_ready) {
                    self.send("session.list", json!({"offset":0,"limit":100}));
                }
            }
            "state.snapshot" => self.apply_snapshot(&frame.data),
            "turn.started" => {
                self.streaming = true;
                self.transcript_auto_follow = true;
                self.active_turn_id = frame.turn_id.unwrap_or_default();
                if let Some(turn) = frame.data.get("turn") {
                    self.transcript.push(TurnView::from_value(turn));
                    trim_vec(&mut self.transcript, MAX_TRANSCRIPT_ITEMS);
                }
                self.status = "Thinking…".into();
            }
            "assistant.delta" => {
                if let Some(turn) = self
                    .transcript
                    .iter_mut()
                    .find(|turn| turn.id == self.active_turn_id)
                {
                    append_bounded(
                        &mut turn.assistant,
                        frame.data.get("text").and_then(Value::as_str).unwrap_or(""),
                        MAX_STREAM_CHARS,
                    );
                }
            }
            "reasoning.delta" => {
                let chunk = frame.data.get("text").and_then(Value::as_str).unwrap_or("");
                append_reasoning(&mut self.transcript, &self.active_turn_id, chunk);
            }
            "tool_call_delta" => {
                update_tool_preview_delta(&mut self.transcript, &self.active_turn_id, &frame.data);
            }
            "tool.requested" => {
                let name = frame
                    .data
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("tool");
                let id = field(&frame.data, "id");
                let stream_id = field(&frame.data, "stream_id");
                let preview = tool_preview_from_request(name, &frame.data);
                append_tool_activity(
                    &mut self.transcript,
                    &self.active_turn_id,
                    &id,
                    &stream_id,
                    name,
                    preview,
                );
                self.status = format!("Running {name}…");
            }
            "tool.completed" => {
                let name = frame
                    .data
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("tool");
                let id = field(&frame.data, "id");
                let preview = tool_preview_from_result(name, &frame.data);
                let failed = frame
                    .data
                    .get("result")
                    .and_then(|result| result.get("ok"))
                    .and_then(Value::as_bool)
                    == Some(false);
                complete_tool_activity(
                    &mut self.transcript,
                    &self.active_turn_id,
                    &id,
                    name,
                    preview,
                    failed,
                );
                self.status = "Working…".into();
            }
            "approval.requested" => {
                let request = frame.data.get("request").unwrap_or(&Value::Null);
                self.approval = Some(Approval {
                    id: frame.approval_id.unwrap_or_default(),
                    reason: field(request, "reason"),
                    target: field(request, "path"),
                    command: field(request, "command"),
                    cwd: field(request, "cwd"),
                });
                self.status = "Action waiting for approval · Y/N".into();
            }
            "turn.completed" => {
                self.approval = None;
                let completed_turn_id = frame
                    .turn_id
                    .as_deref()
                    .filter(|item| !item.is_empty())
                    .unwrap_or(&self.active_turn_id)
                    .to_owned();
                let completed_assistant = self
                    .transcript
                    .iter()
                    .find(|turn| turn.id == completed_turn_id)
                    .map(|turn| turn.assistant.clone())
                    .filter(|text| !text.is_empty())
                    .unwrap_or_else(|| field(&frame.data, "content"));
                let completed_activity = self
                    .transcript
                    .iter()
                    .find(|turn| turn.id == completed_turn_id)
                    .map(|turn| turn.activity.clone())
                    .unwrap_or_default();
                self.status = match frame
                    .data
                    .get("status")
                    .and_then(Value::as_str)
                    .unwrap_or("error")
                {
                    "success" => "Ready".into(),
                    "cancelled" => "Turn cancelled".into(),
                    _ => format!("Turn failed: {}", field(&frame.data, "error")),
                };
                if let Some(snapshot) = frame.data.get("snapshot") {
                    self.apply_snapshot(snapshot);
                }
                // Completion is authoritative. A snapshot may have been produced by
                // the backend worker just before it exited and still report itself
                // as streaming; never let that stale value lock the composer.
                self.streaming = false;
                if !completed_assistant.is_empty() {
                    set_assistant(
                        &mut self.transcript,
                        &completed_turn_id,
                        &completed_assistant,
                    );
                }
                if !completed_activity.is_empty() {
                    set_activity(&mut self.transcript, &completed_turn_id, completed_activity);
                }
                self.active_turn_id.clear();
            }
            "turn.cancellation_acknowledged" => self.status = "Stopping current turn…".into(),
            "session.list" | "session.search" => {
                let items = frame
                    .data
                    .get("items")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                let query = match &self.popup {
                    Some(Popup::Sessions { query, .. }) => query.clone(),
                    _ => String::new(),
                };
                self.popup = Some(Popup::Sessions {
                    query,
                    selected: 0,
                    items,
                });
            }
            "attachments.changed" => {
                self.pending_attachments = frame
                    .data
                    .get("items")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
            }
            "status.changed" => {
                self.model_state = field(&frame.data, "state");
                let model_name = field(&frame.data, "model_name");
                if !model_name.is_empty() {
                    self.model_name = model_name;
                }
                let endpoint = field(&frame.data, "endpoint");
                if !endpoint.is_empty() {
                    self.model_endpoint = endpoint;
                }
                if let Some(window) = frame.data.get("context_window").and_then(Value::as_u64) {
                    self.context_window = Some(window);
                }
            }
            "usage" => self.apply_usage(&frame.data),
            "config.value" => {
                let value = field(&frame.data, "text");
                let cursor = value.len();
                self.popup = Some(Popup::Config { value, cursor });
            }
            "theme.list" => {
                self.themes = frame
                    .data
                    .get("items")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                if let Some(active) = frame.data.get("active") {
                    self.theme = Theme::from_value(active);
                }
            }
            "theme.changed" => {
                if let Some(theme) = frame.data.get("theme") {
                    self.theme = Theme::from_value(theme);
                }
                self.popup = None;
            }
            "palette.items" => {
                self.command_catalog = frame
                    .data
                    .get("items")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
            }
            "skill.changed" => self.send("palette.get", json!({})),
            "command.result" => self.apply_command_result(&frame.data),
            "request.error" | "protocol.error" => {
                self.status = field(&frame.data, "message");
            }
            _ => {}
        }
    }

    fn apply_snapshot(&mut self, value: &Value) {
        let incoming_session_id = value
            .get("session")
            .map(|session| field(session, "id"))
            .unwrap_or_default();
        let session_changed =
            !incoming_session_id.is_empty() && incoming_session_id != self.session_id;
        if session_changed {
            self.context_tokens = None;
            self.command_messages.clear();
            self.transcript_auto_follow = true;
        }
        if let Some(session) = value.get("session") {
            self.session_id = field(session, "id");
            self.session_title = field(session, "title");
            self.branch_name = field(session, "branch_name");
            if self.branch_name.is_empty() {
                self.branch_name = "root".into();
            }
            self.collaboration_mode = field(session, "collaboration_mode");
        }
        if let Some(items) = value.get("transcript").and_then(Value::as_array) {
            let incoming: Vec<TurnView> = items.iter().map(TurnView::from_value).collect();
            let offset = value
                .get("transcript_offset")
                .and_then(Value::as_u64)
                .map(|item| item as usize);
            if !session_changed
                && offset.is_some_and(|item| {
                    self.transcript_offset.is_some_and(|current| item < current)
                })
            {
                let existing: std::collections::HashSet<String> =
                    self.transcript.iter().map(|turn| turn.id.clone()).collect();
                let mut combined: Vec<TurnView> = incoming
                    .into_iter()
                    .filter(|turn| !existing.contains(&turn.id))
                    .collect();
                let prepended = combined.len();
                combined.append(&mut self.transcript);
                let dropped = combined.len().saturating_sub(MAX_TRANSCRIPT_ITEMS);
                self.transcript = combined;
                trim_vec(&mut self.transcript, MAX_TRANSCRIPT_ITEMS);
                let position_shift = prepended.saturating_sub(dropped);
                shift_command_positions(&mut self.command_messages, position_shift);
            } else {
                self.transcript = incoming;
            }
            self.transcript_offset = match (self.transcript_offset, offset) {
                (Some(current), Some(new)) if !session_changed => Some(current.min(new)),
                (_, new) => new,
            };
            self.transcript_previous = value
                .get("transcript_previous")
                .and_then(Value::as_u64)
                .map(|item| item as usize);
        }
        if let Some(items) = value.get("tree").and_then(Value::as_array) {
            let incoming: Vec<TurnView> = items.iter().map(TurnView::from_value).collect();
            let offset = value
                .get("tree_offset")
                .and_then(Value::as_u64)
                .map(|item| item as usize);
            if !session_changed
                && offset.is_some_and(|item| self.tree_offset.is_some_and(|current| item < current))
            {
                let existing: std::collections::HashSet<String> =
                    self.tree.iter().map(|turn| turn.id.clone()).collect();
                let mut combined: Vec<TurnView> = incoming
                    .into_iter()
                    .filter(|turn| !existing.contains(&turn.id))
                    .collect();
                combined.append(&mut self.tree);
                self.tree = combined;
                trim_vec(&mut self.tree, MAX_TRANSCRIPT_ITEMS);
            } else {
                self.tree = incoming;
            }
            self.tree_offset = match (self.tree_offset, offset) {
                (Some(current), Some(new)) if !session_changed => Some(current.min(new)),
                (_, new) => new,
            };
            self.tree_previous = value
                .get("tree_previous")
                .and_then(Value::as_u64)
                .map(|item| item as usize);
            self.tree_selected = self.tree_selected.min(self.tree.len().saturating_sub(1));
        }
        if let Some(items) = value.get("pending_attachments").and_then(Value::as_array) {
            self.pending_attachments = items.clone();
        }
        if let Some(model) = value.get("model") {
            self.model_state = field(model, "state");
            let model_name = field(model, "model_name");
            if !model_name.is_empty() {
                self.model_name = model_name;
            }
            let endpoint = field(model, "endpoint");
            if !endpoint.is_empty() {
                self.model_endpoint = endpoint;
            }
            if let Some(window) = model.get("context_window").and_then(Value::as_u64) {
                self.context_window = Some(window);
            }
        }
        self.streaming = value
            .get("streaming")
            .and_then(Value::as_bool)
            .unwrap_or(self.streaming);
    }

    fn apply_usage(&mut self, value: &Value) {
        let usage = value.get("usage").unwrap_or(value);
        self.context_tokens = ["input_tokens", "prompt_tokens", "prompt_eval_count"]
            .iter()
            .find_map(|key| usage.get(*key).and_then(Value::as_u64))
            .or(self.context_tokens);
    }

    fn apply_command_result(&mut self, value: &Value) {
        let lines = value
            .get("lines")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let mut message = lines
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>()
            .join("\n");
        let action = value.get("action").and_then(Value::as_str).unwrap_or("");
        match action {
            "quit" => self.should_quit = true,
            "help" => {}
            "clear" => {
                self.transcript.clear();
                self.tree.clear();
                self.command_messages.clear();
                self.pending_attachments.clear();
                self.active_turn_id.clear();
                self.transcript_offset = Some(0);
                self.transcript_previous = None;
                self.transcript_scroll = 0;
                self.transcript_max_scroll = 0;
                self.transcript_auto_follow = true;
                self.tree_offset = Some(0);
                self.tree_previous = None;
                self.tree_selected = 0;
            }
            "sessions" => self.send("session.list", json!({"offset":0,"limit":100})),
            "theme" => {
                self.popup = Some(Popup::Theme {
                    selected: 0,
                    items: self.themes.clone(),
                })
            }
            "config" => self.send("config.get", json!({})),
            "health" => {
                self.popup = Some(Popup::Health {
                    report: pretty_json(value.get("report").unwrap_or(&Value::Null)),
                })
            }
            "file_picker" => {
                self.popup = Some(Popup::Palette {
                    query: String::new(),
                    selected: 0,
                    mode: PaletteMode::Files,
                })
            }
            "attach" => self.send("attachment.add", json!({"path": field(value, "path")})),
            "detach" => self.send(
                "attachment.remove",
                json!({"index":value.get("target").cloned().unwrap_or(Value::String("last".into()))}),
            ),
            "toggle_details" => {
                self.show_details = !self.show_details;
                message = format!(
                    "Tool execution details {}",
                    if self.show_details { "shown" } else { "hidden" }
                );
            }
            "toggle_thinking" => {
                self.thinking = !self.thinking;
                message = format!(
                    "Thinking mode {}",
                    if self.thinking { "enabled" } else { "disabled" }
                );
            }
            "code" => {
                self.popup = Some(Popup::Code {
                    content: self
                        .latest_code_block()
                        .unwrap_or_else(|| "No code blocks available".into()),
                })
            }
            _ => {}
        }
        if !message.is_empty() {
            let turn_position = self.transcript.len();
            push_bounded(
                &mut self.command_messages,
                CommandMessage {
                    text: message,
                    turn_position,
                },
                64,
            );
            self.transcript_auto_follow = true;
        }
        if value.get("ok").and_then(Value::as_bool) == Some(false) {
            self.status = lines
                .first()
                .and_then(Value::as_str)
                .unwrap_or("Command failed")
                .into();
        } else if !self.streaming {
            // Successful commands must clear any previous command error.  Keep
            // live turn status intact when commands such as /details are used
            // while a turn is active.
            self.status = "Ready".into();
        }
    }

    fn latest_code_block(&self) -> Option<String> {
        self.transcript.iter().rev().find_map(|turn| {
            let start = turn.assistant.rfind("```")?;
            let before = &turn.assistant[..start];
            let open = before.rfind("```")?;
            let content = &turn.assistant[open + 3..start];
            Some(
                content
                    .trim_start_matches(|character: char| !character.is_whitespace())
                    .trim()
                    .into(),
            )
        })
    }

    fn submit(&mut self) {
        let mut value = self.input.trim().to_owned();
        for chunk in &self.paste_chunks {
            value = value.replace(&chunk.marker, &chunk.text);
        }
        value = strip_terminal_mouse_reports(&value).trim().to_owned();
        if value.is_empty() || !self.connected {
            return;
        }
        if should_retain_draft(self.streaming, &value) {
            self.status = "A turn is already running · draft retained".into();
            return;
        }
        self.input.clear();
        self.cursor = 0;
        self.command_suggestions = false;
        self.command_selected = 0;
        self.paste_chunks.clear();
        if value.starts_with('/') {
            self.send("command.execute", json!({"command":value}));
        } else {
            self.send(
                "turn.start",
                json!({"prompt":value,"thinking":self.thinking}),
            );
        }
    }

    fn handle_event(&mut self, event: Event) {
        match event {
            Event::Key(key) => self.handle_key(key),
            Event::Mouse(mouse) => self.handle_mouse(mouse),
            Event::Paste(text) => self.insert_paste(text),
            Event::Resize(_, _) | Event::FocusGained | Event::FocusLost => {}
        }
    }

    fn handle_key(&mut self, key: KeyEvent) {
        if key.modifiers.contains(KeyModifiers::CONTROL)
            && matches!(key.code, KeyCode::Char('c') | KeyCode::Char('d'))
        {
            self.should_quit = true;
            return;
        }
        if let Some(approval) = self.approval.clone() {
            match key.code {
                KeyCode::Char('y') | KeyCode::Char('Y') => {
                    self.send(
                        "approval.resolve",
                        json!({"approval_id":approval.id,"approved":true}),
                    );
                    self.approval = None;
                }
                KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                    self.send(
                        "approval.resolve",
                        json!({"approval_id":approval.id,"approved":false}),
                    );
                    self.approval = None;
                }
                _ => {}
            }
            return;
        }
        if self.popup.is_some() {
            self.handle_popup_key(key);
            return;
        }
        if key.modifiers.contains(KeyModifiers::CONTROL) {
            match key.code {
                KeyCode::Char('b') => {
                    self.sidebar = !self.sidebar;
                    if !self.sidebar && self.focus == Focus::Tree {
                        self.focus = Focus::Input;
                    }
                }
                KeyCode::Char('f') => {
                    self.open_file_picker();
                }
                KeyCode::Char('g') => self.focus = Focus::Input,
                KeyCode::Char('h') => self.focus = Focus::Transcript,
                KeyCode::Char('l') => {
                    self.sidebar = true;
                    self.focus = Focus::Tree;
                }
                KeyCode::Char('p') if !key.modifiers.contains(KeyModifiers::SHIFT) => {
                    self.popup = Some(Popup::Palette {
                        query: String::new(),
                        selected: 0,
                        mode: PaletteMode::Commands,
                    });
                }
                KeyCode::Char('k') if !key.modifiers.contains(KeyModifiers::SHIFT) => {
                    self.popup = Some(Popup::Palette {
                        query: String::new(),
                        selected: 0,
                        mode: PaletteMode::Global,
                    });
                }
                KeyCode::Char('u') if self.focus == Focus::Input => {
                    self.input.clear();
                    self.cursor = 0;
                }
                KeyCode::Char('k') if self.focus == Focus::Input => {
                    self.input.truncate(self.cursor);
                }
                KeyCode::Char('a') if self.focus == Focus::Input => self.cursor = 0,
                KeyCode::Char('e') if self.focus == Focus::Input => self.cursor = self.input.len(),
                KeyCode::Char('y') if self.focus == Focus::Transcript => {
                    let content = self
                        .transcript
                        .iter()
                        .map(|turn| format!("You:\n{}\n\nAlphanus:\n{}", turn.user, turn.assistant))
                        .collect::<Vec<_>>()
                        .join("\n\n");
                    self.copy_to_clipboard(content);
                }
                KeyCode::Backspace if self.focus == Focus::Input => {
                    self.send(
                        "attachment.remove",
                        json!({"index":if key.modifiers.contains(KeyModifiers::SHIFT) { "all" } else { "last" }}),
                    );
                }
                _ => {}
            }
            return;
        }
        if should_show_help(&key, self.focus) {
            self.show_help();
            return;
        }
        match key.code {
            KeyCode::F(2) => self.show_details = !self.show_details,
            KeyCode::F(3) => self.thinking = !self.thinking,
            KeyCode::Tab if self.focus == Focus::Input && self.command_suggestions => {
                self.complete_selected_command();
            }
            KeyCode::Tab => {
                self.focus = match self.focus {
                    Focus::Input => Focus::Transcript,
                    Focus::Transcript if self.sidebar => Focus::Tree,
                    Focus::Transcript => Focus::Input,
                    Focus::Tree => Focus::Input,
                }
            }
            KeyCode::BackTab => {
                self.focus = match self.focus {
                    Focus::Input if self.sidebar => Focus::Tree,
                    Focus::Input => Focus::Transcript,
                    Focus::Tree => Focus::Transcript,
                    Focus::Transcript => Focus::Input,
                }
            }
            KeyCode::PageUp => {
                self.transcript_scroll = self.transcript_scroll.saturating_sub(8);
                self.transcript_auto_follow = false;
                if let Some(previous) = self.transcript_previous.take() {
                    self.send(
                        "state.get",
                        json!({"transcript_offset":previous,"tree_offset":self.tree_offset}),
                    );
                }
            }
            KeyCode::PageDown => {
                self.transcript_scroll = self
                    .transcript_scroll
                    .saturating_add(8)
                    .min(self.transcript_max_scroll);
                self.transcript_auto_follow = self.transcript_scroll == self.transcript_max_scroll;
            }
            KeyCode::Esc if self.command_suggestions => {
                self.command_suggestions = false;
                self.command_selected = 0;
            }
            KeyCode::Esc => self.handle_escape(),
            KeyCode::Char('/') if self.focus == Focus::Input && self.input.is_empty() => {
                self.edit_input(key);
                self.command_suggestions = true;
                self.command_selected = 0;
            }
            KeyCode::Down if self.focus == Focus::Input && self.command_suggestions => {
                let last = self.command_matches().len().saturating_sub(1);
                self.command_selected = (self.command_selected + 1).min(last);
            }
            KeyCode::Up if self.focus == Focus::Input && self.command_suggestions => {
                self.command_selected = self.command_selected.saturating_sub(1);
            }
            KeyCode::Enter if self.focus == Focus::Input && self.command_suggestions => {
                self.complete_selected_command();
                self.submit();
            }
            KeyCode::Enter if self.focus == Focus::Input => self.submit(),
            KeyCode::Enter | KeyCode::Char('o') if self.focus == Focus::Tree => {
                self.open_selected_tree()
            }
            KeyCode::Char('j') | KeyCode::Down if self.focus == Focus::Tree => {
                self.tree_selected =
                    (self.tree_selected + 1).min(self.tree.len().saturating_sub(1));
            }
            KeyCode::Char('k') | KeyCode::Up if self.focus == Focus::Tree => {
                if self.tree_selected == 0 {
                    if let Some(previous) = self.tree_previous.take() {
                        self.send("state.get", json!({"transcript_offset":self.transcript_offset,"tree_offset":previous}));
                    }
                } else {
                    self.tree_selected = self.tree_selected.saturating_sub(1)
                }
            }
            KeyCode::Char('g') if self.focus == Focus::Tree => self.tree_selected = 0,
            KeyCode::Char('G') if self.focus == Focus::Tree => {
                self.tree_selected = self.tree.len().saturating_sub(1)
            }
            KeyCode::Char('[') if self.focus == Focus::Tree => self.move_tree_sibling(-1),
            KeyCode::Char(']') if self.focus == Focus::Tree => self.move_tree_sibling(1),
            _ if self.focus == Focus::Input => {
                self.edit_input(key);
                self.refresh_command_suggestions();
            }
            _ => {}
        }
    }

    fn show_help(&mut self) {
        let turn_position = self.transcript.len();
        push_bounded(
            &mut self.command_messages,
            CommandMessage {
                text: catalog_text(&self.shortcut_catalog, "key"),
                turn_position,
            },
            64,
        );
        self.transcript_auto_follow = true;
    }

    fn open_file_picker(&mut self) {
        if self.streaming {
            return;
        }
        self.popup = Some(Popup::Palette {
            query: String::new(),
            selected: 0,
            mode: PaletteMode::Files,
        });
    }

    fn command_matches(&self) -> Vec<Value> {
        let query = self.input.strip_prefix('/').unwrap_or(&self.input);
        filtered_palette(&self.command_catalog, query, PaletteMode::Commands)
    }

    fn refresh_command_suggestions(&mut self) {
        self.command_suggestions =
            self.input.starts_with('/') && !self.input[1..].chars().any(char::is_whitespace);
        let last = self.command_matches().len().saturating_sub(1);
        self.command_selected = self.command_selected.min(last);
    }

    fn complete_selected_command(&mut self) {
        if let Some(item) = self.command_matches().get(self.command_selected) {
            self.input = palette_value(item);
            self.cursor = self.input.len();
        }
        self.command_suggestions = false;
        self.command_selected = 0;
    }

    fn edit_input(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char(character) => {
                self.input.insert(self.cursor, character);
                self.cursor += character.len_utf8();
            }
            KeyCode::Backspace if self.cursor > 0 => {
                let previous = self.input[..self.cursor]
                    .char_indices()
                    .next_back()
                    .map(|(index, _)| index)
                    .unwrap_or(0);
                self.input.drain(previous..self.cursor);
                self.cursor = previous;
            }
            KeyCode::Delete if self.cursor < self.input.len() => {
                let next = self.input[self.cursor..]
                    .char_indices()
                    .nth(1)
                    .map(|(index, _)| self.cursor + index)
                    .unwrap_or(self.input.len());
                self.input.drain(self.cursor..next);
            }
            KeyCode::Left => {
                self.cursor = self.input[..self.cursor]
                    .char_indices()
                    .next_back()
                    .map(|(index, _)| index)
                    .unwrap_or(0)
            }
            KeyCode::Right => {
                self.cursor = self.input[self.cursor..]
                    .char_indices()
                    .nth(1)
                    .map(|(index, _)| self.cursor + index)
                    .unwrap_or(self.input.len())
            }
            KeyCode::Home => self.cursor = 0,
            KeyCode::End => self.cursor = self.input.len(),
            KeyCode::Backspace if self.input.is_empty() => {
                self.send("attachment.remove", json!({"index":"last"}));
            }
            _ => {}
        }
    }

    fn insert_paste(&mut self, text: String) {
        if text.len() < PASTE_THRESHOLD {
            self.input.insert_str(self.cursor, &text);
            self.cursor += text.len();
            self.refresh_command_suggestions();
            return;
        }
        let marker = format!("[Pasted {} chars]", text.chars().count());
        self.input.insert_str(self.cursor, &marker);
        self.cursor += marker.len();
        self.paste_chunks.push(PasteChunk { marker, text });
        self.refresh_command_suggestions();
    }

    fn handle_escape(&mut self) {
        if self.streaming {
            let now = Instant::now();
            if self
                .last_escape
                .is_some_and(|last| now.duration_since(last) <= CANCEL_WINDOW)
            {
                self.send("turn.cancel", json!({}));
                self.last_escape = None;
            } else {
                self.last_escape = Some(now);
                self.status = "Press Esc again to stop the current turn".into();
            }
        } else {
            self.input.clear();
            self.cursor = 0;
            self.command_suggestions = false;
            self.command_selected = 0;
            self.paste_chunks.clear();
        }
    }

    fn handle_popup_key(&mut self, key: KeyEvent) {
        if key.code == KeyCode::Esc {
            self.popup = None;
            return;
        }
        let mut deferred: Option<(&'static str, Value)> = None;
        let mut clipboard: Option<String> = None;
        match self.popup.as_mut() {
            Some(Popup::Palette {
                query,
                selected,
                mode,
            }) => match key.code {
                KeyCode::Char(character) => query.push(character),
                KeyCode::Backspace => {
                    query.pop();
                }
                KeyCode::Down => *selected = selected.saturating_add(1),
                KeyCode::Up => *selected = selected.saturating_sub(1),
                KeyCode::Enter => {
                    let items = filtered_palette(&self.command_catalog, query, *mode);
                    if let Some(item) = items.get(*selected) {
                        deferred = palette_request(item);
                        if field(item, "kind") == "command" {
                            self.input = palette_value(item);
                            self.cursor = self.input.len();
                        }
                    }
                    self.popup = None;
                }
                _ => {}
            },
            Some(Popup::Sessions {
                query,
                selected,
                items,
            }) => match key.code {
                KeyCode::Char('n') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    self.popup = Some(Popup::SessionName {
                        value: String::new(),
                    });
                }
                KeyCode::Char(character) => {
                    query.push(character);
                    deferred = Some(("session.search", json!({"query":query.clone(),"limit":80})));
                }
                KeyCode::Backspace => {
                    query.pop();
                    deferred = Some(if query.is_empty() {
                        ("session.list", json!({"offset":0,"limit":100}))
                    } else {
                        ("session.search", json!({"query":query.clone(),"limit":80}))
                    });
                }
                KeyCode::Down => *selected = (*selected + 1).min(items.len().saturating_sub(1)),
                KeyCode::Up => *selected = selected.saturating_sub(1),
                KeyCode::Enter => {
                    if let Some(item) = items.get(*selected) {
                        deferred = Some(("session.load", session_load_data(item)));
                    }
                    self.popup = None;
                }
                KeyCode::Delete => {
                    if let Some(item) = items.get(*selected) {
                        let id = session_identity(item);
                        if self.session_delete_armed.as_deref() == Some(id) {
                            deferred = Some(("session.delete", json!({"session_id":id})));
                            self.session_delete_armed = None;
                            self.popup = None;
                        } else {
                            self.session_delete_armed = Some(id.to_owned());
                            self.status = "Press Delete again to confirm session removal".into();
                        }
                    }
                }
                _ => {}
            },
            Some(Popup::SessionName { value }) => match key.code {
                KeyCode::Char(character) => value.push(character),
                KeyCode::Backspace => {
                    value.pop();
                }
                KeyCode::Enter => {
                    deferred = Some(("session.create", json!({"title":value.clone()})));
                    self.popup = None;
                }
                _ => {}
            },
            Some(Popup::Theme { selected, items }) => match key.code {
                KeyCode::Down => *selected = (*selected + 1).min(items.len().saturating_sub(1)),
                KeyCode::Up => *selected = selected.saturating_sub(1),
                KeyCode::Enter => {
                    if let Some(id) = items
                        .get(*selected)
                        .and_then(|value| value.get("id"))
                        .and_then(Value::as_str)
                    {
                        deferred = Some(("theme.apply", json!({"theme_id":id})));
                    }
                }
                _ => {}
            },
            Some(Popup::Config { value, cursor }) => match key.code {
                KeyCode::Char('s') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    deferred = Some(("config.apply", json!({"text":value.clone()})));
                    self.popup = None;
                }
                KeyCode::Char('y') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    clipboard = Some(value.clone())
                }
                KeyCode::Char(character) => {
                    value.insert(*cursor, character);
                    *cursor += character.len_utf8();
                }
                KeyCode::Enter => {
                    value.insert(*cursor, '\n');
                    *cursor += 1;
                }
                KeyCode::Backspace if *cursor > 0 => {
                    let previous = value[..*cursor]
                        .char_indices()
                        .next_back()
                        .map(|(index, _)| index)
                        .unwrap_or(0);
                    value.drain(previous..*cursor);
                    *cursor = previous;
                }
                KeyCode::Left => {
                    *cursor = value[..*cursor]
                        .char_indices()
                        .next_back()
                        .map(|(index, _)| index)
                        .unwrap_or(0)
                }
                KeyCode::Right => {
                    *cursor = value[*cursor..]
                        .char_indices()
                        .nth(1)
                        .map(|(index, _)| *cursor + index)
                        .unwrap_or(value.len())
                }
                _ => {}
            },
            Some(Popup::Code { content }) => {
                if matches!(key.code, KeyCode::Char('y') | KeyCode::Char('c')) {
                    clipboard = Some(content.clone());
                }
            }
            Some(Popup::Fatal) => match key.code {
                KeyCode::Char('r') => self.restart_backend(),
                KeyCode::Char('q') => self.should_quit = true,
                _ => {}
            },
            Some(Popup::Health { .. }) | None => {}
        }
        if let Some((kind, data)) = deferred {
            self.send(kind, data);
        }
        if let Some(content) = clipboard {
            self.copy_to_clipboard(content);
        }
    }

    fn handle_mouse(&mut self, mouse: MouseEvent) {
        if self.approval.is_some() {
            if matches!(mouse.kind, MouseEventKind::Down(MouseButton::Left)) {
                let position = Position::new(mouse.column, mouse.row);
                let approved = self.approval_approve_area.contains(position);
                let denied = self.approval_deny_area.contains(position);
                if !approved && !denied {
                    return;
                }
                if let Some(approval) = self.approval.take() {
                    self.send(
                        "approval.resolve",
                        json!({"approval_id":approval.id,"approved":approved}),
                    );
                }
            }
            return;
        }
        if self.popup.is_some() {
            self.handle_popup_mouse(mouse);
            return;
        }
        if self.command_suggestions
            && self
                .command_suggestions_area
                .contains(Position::new(mouse.column, mouse.row))
        {
            match mouse.kind {
                MouseEventKind::ScrollUp => {
                    let last = self.command_matches().len().saturating_sub(1);
                    self.command_selected = (self.command_selected + 1).min(last);
                }
                MouseEventKind::ScrollDown => {
                    self.command_selected = self.command_selected.saturating_sub(1)
                }
                MouseEventKind::Down(MouseButton::Left) => {
                    let row = mouse
                        .row
                        .saturating_sub(self.command_suggestions_area.y + 1)
                        as usize;
                    let start = self.command_selected.saturating_sub(7);
                    if start + row < self.command_matches().len() {
                        self.command_selected = start + row;
                        self.complete_selected_command();
                    }
                }
                _ => {}
            }
            return;
        }
        match mouse.kind {
            MouseEventKind::ScrollUp => {
                if self
                    .sidebar_area
                    .contains(Position::new(mouse.column, mouse.row))
                {
                    self.tree_selected =
                        (self.tree_selected + 1).min(self.tree.len().saturating_sub(1));
                } else {
                    self.transcript_scroll =
                        touchpad_scroll(self.transcript_scroll, true, self.transcript_max_scroll);
                    self.transcript_auto_follow =
                        self.transcript_scroll == self.transcript_max_scroll;
                }
            }
            MouseEventKind::ScrollDown => {
                if self
                    .sidebar_area
                    .contains(Position::new(mouse.column, mouse.row))
                {
                    self.tree_selected = self.tree_selected.saturating_sub(1);
                } else {
                    self.transcript_scroll =
                        touchpad_scroll(self.transcript_scroll, false, self.transcript_max_scroll);
                    self.transcript_auto_follow =
                        self.transcript_scroll == self.transcript_max_scroll;
                }
            }
            MouseEventKind::Down(MouseButton::Left) => {
                let position = Position::new(mouse.column, mouse.row);
                if self.file_button_area.contains(position) {
                    self.open_file_picker();
                } else if self.input_area.contains(position) {
                    self.focus = Focus::Input;
                } else if self.sidebar_area.contains(position) {
                    self.command_suggestions = false;
                    self.focus = Focus::Tree;
                    let relative = mouse.row.saturating_sub(self.sidebar_area.y + 3) as usize;
                    self.tree_selected = relative.min(self.tree.len().saturating_sub(1));
                } else if self.transcript_area.contains(position) {
                    self.command_suggestions = false;
                    self.focus = Focus::Transcript;
                }
            }
            MouseEventKind::Up(MouseButton::Left)
            | MouseEventKind::Drag(MouseButton::Left)
            | MouseEventKind::Moved => {}
            _ => {}
        }
    }

    fn handle_popup_mouse(&mut self, mouse: MouseEvent) {
        let position = Position::new(mouse.column, mouse.row);
        if !self.popup_area.contains(position) {
            return;
        }
        match mouse.kind {
            MouseEventKind::ScrollUp => {
                self.handle_popup_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE))
            }
            MouseEventKind::ScrollDown => {
                self.handle_popup_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
            }
            MouseEventKind::Down(MouseButton::Left) => {
                let relative_row = mouse.row.saturating_sub(self.popup_area.y) as usize;
                let relative_column = mouse.column.saturating_sub(self.popup_area.x);
                match self.popup.clone() {
                    Some(Popup::Sessions { .. }) if self.popup_new_area.contains(position) => {
                        self.popup = Some(Popup::SessionName {
                            value: String::new(),
                        });
                    }
                    Some(Popup::Sessions { items, .. })
                        if self.popup_list_area.contains(position) =>
                    {
                        let selected = self.popup_list_offset
                            + mouse.row.saturating_sub(self.popup_list_area.y) as usize;
                        if let Some(item) = items.get(selected) {
                            let data = session_load_data(item);
                            self.popup = None;
                            self.send("session.load", data);
                        }
                    }
                    Some(Popup::Palette { query, mode, .. }) if relative_row >= 3 => {
                        let selected = relative_row - 3;
                        let items = filtered_palette(&self.command_catalog, &query, mode);
                        if let Some(item) = items.get(selected) {
                            let request = palette_request(item);
                            if field(item, "kind") == "command" {
                                self.input = palette_value(item);
                                self.cursor = self.input.len();
                            }
                            self.popup = None;
                            if let Some((kind, data)) = request {
                                self.send(kind, data);
                            }
                        }
                    }
                    Some(Popup::Theme { items, .. }) if relative_row >= 1 => {
                        if let Some(id) = items
                            .get(relative_row - 1)
                            .and_then(|value| value.get("id"))
                            .and_then(Value::as_str)
                        {
                            let id = id.to_owned();
                            self.send("theme.apply", json!({"theme_id":id}));
                        }
                    }
                    Some(Popup::SessionName { value })
                        if relative_row + 3 >= self.popup_area.height as usize =>
                    {
                        if relative_column < self.popup_area.width / 2 {
                            self.popup = None;
                            self.send("session.create", json!({"title":value}));
                        } else {
                            self.popup = None;
                        }
                    }
                    Some(Popup::Config { value, .. })
                        if relative_row + 3 >= self.popup_area.height as usize =>
                    {
                        if relative_column < self.popup_area.width / 2 {
                            self.popup = None;
                            self.send("config.apply", json!({"text":value}));
                        } else {
                            self.popup = None;
                        }
                    }
                    Some(Popup::Code { content }) if relative_row <= 2 => {
                        self.copy_to_clipboard(content)
                    }
                    Some(Popup::Fatal) if relative_row + 3 >= self.popup_area.height as usize => {
                        if relative_column < self.popup_area.width / 2 {
                            self.restart_backend();
                        } else {
                            self.should_quit = true;
                        }
                    }
                    Some(Popup::Health { .. } | Popup::Code { .. }) => self.popup = None,
                    _ => {}
                }
            }
            _ => {}
        }
    }

    fn open_selected_tree(&mut self) {
        if let Some(turn) = self.tree.get(self.tree_selected) {
            self.send("branch.open", json!({"turn_id":turn.id}));
        }
    }

    fn move_tree_sibling(&mut self, direction: isize) {
        let Some(current) = self.tree.get(self.tree_selected) else {
            return;
        };
        let siblings: Vec<usize> = self
            .tree
            .iter()
            .enumerate()
            .filter_map(|(index, turn)| (turn.parent == current.parent).then_some(index))
            .collect();
        let Some(position) = siblings
            .iter()
            .position(|index| *index == self.tree_selected)
        else {
            return;
        };
        let target = if direction < 0 {
            position.saturating_sub(1)
        } else {
            (position + 1).min(siblings.len().saturating_sub(1))
        };
        if let Some(index) = siblings.get(target) {
            self.tree_selected = *index;
        }
    }

    fn copy_to_clipboard(&mut self, content: String) {
        let payload = BASE64.encode(content.as_bytes());
        let result = write!(stdout(), "\x1b]52;c;{payload}\x07").and_then(|_| stdout().flush());
        self.clipboard_notice = Some((
            if result.is_ok() {
                "Copied with OSC 52".into()
            } else {
                "Clipboard unavailable; use Shift-drag".into()
            },
            Instant::now(),
        ));
    }
}

struct TerminalGuard;

impl TerminalGuard {
    fn enter() -> Result<Self, String> {
        enable_raw_mode().map_err(|error| error.to_string())?;
        execute!(
            stdout(),
            EnterAlternateScreen,
            EnableMouseCapture,
            EnableBracketedPaste
        )
        .map_err(|error| error.to_string())?;
        Ok(Self)
    }
}

struct SignalGuard {
    requested: Arc<AtomicBool>,
    registrations: Vec<signal_hook::SigId>,
}

impl SignalGuard {
    fn install() -> Result<Self, String> {
        let requested = Arc::new(AtomicBool::new(false));
        let mut registrations = Vec::with_capacity(2);
        for signal in [signal_hook::consts::SIGINT, signal_hook::consts::SIGTERM] {
            match signal_hook::flag::register(signal, Arc::clone(&requested)) {
                Ok(registration) => registrations.push(registration),
                Err(error) => {
                    for registration in registrations {
                        signal_hook::low_level::unregister(registration);
                    }
                    return Err(format!("failed to install signal handler: {error}"));
                }
            }
        }
        Ok(Self {
            requested,
            registrations,
        })
    }

    fn requested(&self) -> bool {
        self.requested.load(Ordering::Relaxed)
    }
}

impl Drop for SignalGuard {
    fn drop(&mut self) {
        for registration in self.registrations.drain(..) {
            signal_hook::low_level::unregister(registration);
        }
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(
            stdout(),
            DisableBracketedPaste,
            DisableMouseCapture,
            LeaveAlternateScreen,
            Show
        );
    }
}

pub fn run(python: &str, project_root: Option<&str>, debug: bool) -> Result<i32, String> {
    let mut app = App::new(python, project_root, debug)?;
    let _guard = TerminalGuard::enter()?;
    let signals = SignalGuard::install()?;
    let backend = CrosstermBackend::new(stdout());
    let mut terminal = Terminal::new(backend).map_err(|error| error.to_string())?;
    terminal.clear().map_err(|error| error.to_string())?;
    let mut forced_repaints = 0_u8;
    while !app.should_quit && !signals.requested() {
        let fatal_was_visible = matches!(app.popup.as_ref(), Some(Popup::Fatal));
        app.drain_backend();
        if !fatal_was_visible && matches!(app.popup.as_ref(), Some(Popup::Fatal)) {
            // A backend exit can arrive between terminal paint cycles. Force a
            // handful of complete frames so integrated terminals never expose
            // a partially repainted fatal overlay.
            forced_repaints = 4;
            app.last_frame = Instant::now() - Duration::from_millis(16);
        }
        if app.last_frame.elapsed() >= Duration::from_millis(16) {
            if forced_repaints > 0 {
                terminal.clear().map_err(|error| error.to_string())?;
                forced_repaints -= 1;
            }
            // Keep large transcript and popup transitions atomic on terminals
            // that implement synchronized updates. Unsupported terminals safely
            // ignore these ANSI sequences.
            execute!(terminal.backend_mut(), BeginSynchronizedUpdate)
                .map_err(|error| error.to_string())?;
            let draw_result = terminal.draw(|frame| draw(frame, &mut app)).map(|_| ());
            let sync_result = execute!(terminal.backend_mut(), EndSynchronizedUpdate);
            draw_result.map_err(|error| error.to_string())?;
            sync_result.map_err(|error| error.to_string())?;
            app.last_frame = Instant::now();
        }
        if event::poll(Duration::from_millis(8)).map_err(|error| error.to_string())? {
            let next = event::read().map_err(|error| error.to_string())?;
            let terminal_needs_repaint = terminal_repaint_shortcut(&next);
            app.handle_event(next);
            if terminal_needs_repaint {
                // Integrated terminals such as VSCodium may clear their visible
                // buffer for Ctrl+K/Ctrl+L even while forwarding the key to the
                // TUI. OSC 52 clipboard writes can similarly race a terminal
                // compositor. Invalidate Ratatui's back buffer for a few frames
                // instead of repainting only once.
                forced_repaints = 4;
                app.last_frame = Instant::now() - Duration::from_millis(16);
            }
        }
    }
    app.backend.shutdown();
    Ok(0)
}

fn terminal_repaint_shortcut(event: &Event) -> bool {
    matches!(
        event,
        Event::Key(KeyEvent {
            code: KeyCode::Char('k' | 'l' | 'y'),
            modifiers,
            ..
        }) if modifiers.contains(KeyModifiers::CONTROL)
    )
}

fn draw(frame: &mut Frame, app: &mut App) {
    let area = frame.area();
    frame.render_widget(
        Block::default().style(Style::default().bg(app.theme.background)),
        area,
    );
    let horizontal = if app.sidebar && area.width >= 90 {
        Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Min(40), Constraint::Length(38)])
            .split(area)
    } else {
        Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Min(1), Constraint::Length(0)])
            .split(area)
    };
    let main = horizontal[0];
    let sidebar = horizontal[1];
    let approval_height = app
        .approval
        .as_ref()
        .map(|approval| approval_height(main.width, approval))
        .unwrap_or(0);
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(5),
            Constraint::Length(1),
            Constraint::Length(approval_height),
            Constraint::Length(3),
            Constraint::Length(2),
        ])
        .split(main);
    app.transcript_area = vertical[0];
    app.input_area = vertical[3];
    app.sidebar_area = sidebar;
    draw_transcript(frame, app, vertical[0]);
    draw_attachments(frame, app, vertical[1]);
    if let Some(approval) = app.approval.clone() {
        draw_approval(frame, app, &approval, vertical[2]);
    } else {
        app.approval_area = Rect::default();
        app.approval_approve_area = Rect::default();
        app.approval_deny_area = Rect::default();
    }
    draw_input(frame, app, vertical[3]);
    draw_status(frame, app, vertical[4]);
    if sidebar.width > 0 {
        draw_sidebar(frame, app, sidebar);
    }
    if app.command_suggestions && app.popup.is_none() && app.approval.is_none() {
        draw_command_suggestions(frame, app);
    } else {
        app.command_suggestions_area = Rect::default();
    }
    if app.popup.is_some() {
        draw_popup(frame, app);
    }
    if app.focus == Focus::Input && app.popup.is_none() && app.approval.is_none() {
        frame.set_cursor_position(Position::new(
            composer_cursor_x(&app.input, app.cursor, app.composer_area),
            app.input_area.y + 1,
        ));
    }
}

fn draw_command_suggestions(frame: &mut Frame, app: &mut App) {
    let matches = app.command_matches();
    let visible_count = matches
        .len()
        .min(8)
        .min(app.input_area.y.saturating_sub(2) as usize);
    if visible_count == 0 || app.input_area.width < 12 {
        app.command_suggestions_area = Rect::default();
        return;
    }
    let height = visible_count as u16 + 2;
    let x = app.input_area.x.saturating_add(3);
    let available_width = app.input_area.right().saturating_sub(x + 1);
    let width = available_width.min(60);
    if width < 12 {
        app.command_suggestions_area = Rect::default();
        return;
    }
    let area = Rect::new(x, app.input_area.y.saturating_sub(height), width, height);
    app.command_suggestions_area = area;
    let start = app.command_selected.saturating_sub(7);
    let items = matches
        .iter()
        .enumerate()
        .skip(start)
        .take(visible_count)
        .map(|(index, value)| {
            let line = Line::from(vec![
                Span::styled(
                    format!("{:<20}", palette_prompt(value)),
                    Style::default()
                        .fg(app.theme.accent)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    field(value, "description"),
                    Style::default().fg(app.theme.muted),
                ),
            ]);
            ListItem::new(line).style(if index == app.command_selected {
                app.theme.selected()
            } else {
                app.theme.base()
            })
        })
        .collect::<Vec<_>>();
    frame.render_widget(Clear, area);
    frame.render_widget(
        List::new(items).block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(app.theme.border_type())
                .border_style(Style::default().fg(app.theme.accent))
                .title(" Commands ")
                .style(app.theme.base()),
        ),
        area,
    );
}

fn composer_view(input: &str, cursor: usize, area: Rect) -> (String, u16) {
    const PROMPT_WIDTH: usize = 2;
    let cursor = cursor.min(input.len());
    let available = usize::from(area.width.saturating_sub(3).max(1));
    let before = &input[..cursor];
    let mut start = 0;
    if UnicodeWidthStr::width(before) >= available {
        let target_width = available.saturating_sub(1);
        let mut width = 0;
        start = cursor;
        for (index, character) in before.char_indices().rev() {
            let character_width = UnicodeWidthChar::width(character).unwrap_or(0);
            if width + character_width > target_width {
                break;
            }
            width += character_width;
            start = index;
        }
    }

    let mut visible = String::new();
    let mut visible_width = 0;
    for character in input[start..].chars() {
        let character_width = UnicodeWidthChar::width(character).unwrap_or(0);
        if visible_width + character_width > available {
            break;
        }
        visible.push(character);
        visible_width += character_width;
    }
    let cursor_width = UnicodeWidthStr::width(&input[start..cursor]);
    let x = area.x + 1 + (PROMPT_WIDTH + cursor_width) as u16;
    (visible, x.min(area.right().saturating_sub(1)))
}

fn composer_cursor_x(input: &str, cursor: usize, area: Rect) -> u16 {
    composer_view(input, cursor, area).1
}

fn should_show_help(key: &KeyEvent, focus: Focus) -> bool {
    matches!(key.code, KeyCode::F(1))
        || (focus != Focus::Input && matches!(key.code, KeyCode::Char('?')))
}

fn touchpad_scroll(current: u16, scroll_up_event: bool, max_scroll: u16) -> u16 {
    if scroll_up_event {
        current.saturating_sub(3)
    } else {
        current.saturating_add(3).min(max_scroll)
    }
}

fn resolved_transcript_scroll(current: u16, max_scroll: u16, auto_follow: bool) -> u16 {
    if auto_follow {
        max_scroll
    } else {
        current.min(max_scroll)
    }
}

fn take_startup_session_manager(show_on_ready: &mut bool) -> bool {
    std::mem::take(show_on_ready)
}

fn transcript_scrollbar_state(
    current: u16,
    max_scroll: u16,
    viewport_height: usize,
) -> ScrollbarState {
    // Ratatui models `content_length` as the number of valid scrollbar
    // positions, while `viewport_content_length` controls the thumb size.
    // There are max_scroll + 1 valid offsets from the first line to the last.
    ScrollbarState::new(usize::from(max_scroll).saturating_add(1))
        .position(usize::from(current.min(max_scroll)))
        .viewport_content_length(viewport_height.max(1))
}

fn line_is_empty(line: &Line<'_>) -> bool {
    line.spans.iter().all(|span| span.content.trim().is_empty())
}

fn trim_message_lines(mut lines: Vec<Line<'static>>) -> Vec<Line<'static>> {
    while lines.first().is_some_and(line_is_empty) {
        lines.remove(0);
    }
    while lines.last().is_some_and(line_is_empty) {
        lines.pop();
    }
    lines
}

fn dim_message_lines(mut lines: Vec<Line<'static>>) -> Vec<Line<'static>> {
    for line in &mut lines {
        for span in &mut line.spans {
            span.style = span.style.add_modifier(Modifier::DIM);
        }
    }
    lines
}

fn push_spaced_message_block(destination: &mut Vec<Line<'static>>, block: Vec<Line<'static>>) {
    let block = trim_message_lines(block);
    if block.is_empty() {
        return;
    }
    if !destination.is_empty() {
        destination.push(Line::default());
    }
    destination.extend(block);
}

fn rail_line(color: ratatui::style::Color) -> Line<'static> {
    Line::from(vec![
        Span::styled("┃", Style::default().fg(color).add_modifier(Modifier::BOLD)),
        Span::raw(" "),
    ])
}

fn push_styled_character(line: &mut Line<'static>, character: char, style: Style) {
    if let Some(last) = line.spans.last_mut().filter(|span| span.style == style) {
        last.content.to_mut().push(character);
    } else {
        line.spans.push(Span::styled(character.to_string(), style));
    }
}

fn wrap_styled_characters(
    characters: &[(char, Style, usize)],
    available: usize,
    color: ratatui::style::Color,
) -> Vec<Line<'static>> {
    wrap_styled_characters_with_prefix(characters, available, &rail_line(color))
}

fn wrap_styled_characters_with_prefix(
    characters: &[(char, Style, usize)],
    available: usize,
    prefix: &Line<'static>,
) -> Vec<Line<'static>> {
    if characters.is_empty() {
        return vec![prefix.clone()];
    }
    let mut output = Vec::new();
    let mut start = 0;
    while start < characters.len() {
        let mut width = 0;
        let mut fit_end = start;
        while fit_end < characters.len() && width + characters[fit_end].2 <= available {
            width += characters[fit_end].2;
            fit_end += 1;
        }
        if fit_end == start {
            fit_end += 1;
        }
        let mut end = fit_end;
        let mut next = fit_end;
        if fit_end < characters.len() {
            if let Some(space) = (start + 1..fit_end)
                .rev()
                .find(|index| characters[*index].0.is_whitespace())
            {
                end = space;
                next = space + 1;
                while next < characters.len() && characters[next].0.is_whitespace() {
                    next += 1;
                }
            }
        }
        let mut line = prefix.clone();
        for (character, style, _) in &characters[start..end] {
            push_styled_character(&mut line, *character, *style);
        }
        output.push(line);
        start = next;
    }
    output
}

fn wrap_styled_lines(
    source: Vec<Line<'static>>,
    width: usize,
    prefix: &Line<'static>,
) -> Vec<Line<'static>> {
    let mut output = Vec::new();
    for source_line in source {
        let mut characters = Vec::new();
        for span in source_line.spans {
            for character in span.content.chars() {
                if character == '\n' {
                    output.extend(wrap_styled_characters_with_prefix(
                        &characters,
                        width,
                        prefix,
                    ));
                    characters.clear();
                    continue;
                }
                characters.push((
                    character,
                    span.style,
                    UnicodeWidthChar::width(character).unwrap_or(0),
                ));
            }
        }
        output.extend(wrap_styled_characters_with_prefix(
            &characters,
            width,
            prefix,
        ));
    }
    output
}

fn preview_box_lines(
    source: Vec<Line<'static>>,
    available: usize,
    border: ratatui::style::Color,
    background: ratatui::style::Color,
) -> Vec<Line<'static>> {
    if available < 4 {
        return source;
    }
    let widest = source
        .iter()
        .map(|line| {
            line.spans
                .iter()
                .map(|span| UnicodeWidthStr::width(span.content.as_ref()))
                .sum::<usize>()
        })
        .max()
        .unwrap_or(1);
    let max_width = available.min(96);
    let box_width = widest
        .saturating_add(2)
        .max(available.min(36))
        .min(max_width);
    let inner_width = box_width.saturating_sub(2).max(1);
    let border_style = Style::default().fg(border).bg(background);
    let fill_style = Style::default().bg(background);
    let mut output = vec![Line::from(Span::styled(
        format!("┌{}┐", "─".repeat(inner_width)),
        border_style,
    ))];
    for mut line in wrap_styled_lines(source, inner_width, &Line::default()) {
        for span in &mut line.spans {
            span.style = span.style.bg(background);
        }
        let content_width = line
            .spans
            .iter()
            .map(|span| UnicodeWidthStr::width(span.content.as_ref()))
            .sum::<usize>();
        let mut boxed = Line::from(Span::styled("│", border_style));
        boxed.spans.extend(line.spans);
        boxed.spans.push(Span::styled(
            " ".repeat(inner_width.saturating_sub(content_width)),
            fill_style,
        ));
        boxed.spans.push(Span::styled("│", border_style));
        output.push(boxed);
    }
    output.push(Line::from(Span::styled(
        format!("└{}┘", "─".repeat(inner_width)),
        border_style,
    )));
    output
}

fn message_rail_lines(
    source: Vec<Line<'static>>,
    width: u16,
    color: ratatui::style::Color,
) -> Vec<Line<'static>> {
    let available = usize::from(width.saturating_sub(2).max(1));
    let mut output = Vec::new();
    for source_line in source {
        let mut characters = Vec::new();
        for span in source_line.spans {
            for character in span.content.chars() {
                if character == '\n' {
                    output.extend(wrap_styled_characters(&characters, available, color));
                    characters.clear();
                    continue;
                }
                let character_width = UnicodeWidthChar::width(character).unwrap_or(0);
                characters.push((character, span.style, character_width));
            }
        }
        output.extend(wrap_styled_characters(&characters, available, color));
    }
    output
}

fn append_command_lines(
    lines: &mut Vec<Line<'static>>,
    message: &CommandMessage,
    app: &App,
    content_width: u16,
) {
    let command_lines = trim_message_lines(markdown_lines(
        &message.text,
        app.theme.text,
        app.theme.accent,
        &app.theme.syntax_theme,
    ));
    lines.extend(message_rail_lines(
        command_lines,
        content_width,
        app.theme.accent,
    ));
    lines.push(Line::default());
}

fn command_display_position(message: &CommandMessage, transcript_len: usize) -> usize {
    message.turn_position.min(transcript_len)
}

fn shift_command_positions(messages: &mut VecDeque<CommandMessage>, prepended_turns: usize) {
    for message in messages {
        message.turn_position = message.turn_position.saturating_add(prepended_turns);
    }
}

fn draw_transcript(frame: &mut Frame, app: &mut App, area: Rect) {
    let border = if app.focus == Focus::Transcript {
        app.theme.secondary
    } else {
        app.theme.border
    };
    let content_width = area.width.saturating_sub(2).max(1);
    let mut lines = Vec::<Line<'static>>::new();
    for (turn_index, turn) in app.transcript.iter().enumerate() {
        for message in app
            .command_messages
            .iter()
            .filter(|message| command_display_position(message, app.transcript.len()) == turn_index)
        {
            append_command_lines(&mut lines, message, app, content_width);
        }
        let mut user_lines = Vec::new();
        if !turn.attachments.is_empty() {
            user_lines.push(Line::from(Span::styled(
                format!("Attachments: {}", turn.attachments),
                Style::default().fg(app.theme.muted),
            )));
        }
        user_lines.extend(trim_message_lines(markdown_lines(
            &turn.user,
            app.theme.text,
            app.theme.accent,
            &app.theme.syntax_theme,
        )));
        lines.extend(message_rail_lines(
            user_lines,
            content_width,
            app.theme.success,
        ));
        lines.push(Line::default());
        let mut assistant_lines = Vec::new();
        for activity in &turn.activity {
            if activity.kind == "reasoning" && app.show_details && !activity.text.trim().is_empty()
            {
                push_spaced_message_block(
                    &mut assistant_lines,
                    dim_message_lines(markdown_lines(
                        &activity.text,
                        app.theme.muted,
                        app.theme.subtle,
                        &app.theme.syntax_theme,
                    )),
                );
            } else if activity.kind == "tool" {
                let mut tool_lines = Vec::new();
                let (marker, color) = if activity.tool.failed {
                    ("✗", app.theme.error)
                } else if activity.tool.completed {
                    ("✓", app.theme.success)
                } else {
                    ("▶", app.theme.warning)
                };
                tool_lines.push(Line::from(vec![
                    Span::styled(format!("{marker} "), Style::default().fg(color)),
                    Span::styled(
                        activity.tool.name.clone(),
                        Style::default().fg(app.theme.muted),
                    ),
                ]));
                if app.show_details && !activity.tool.preview.is_empty() {
                    let preview_label = if activity.tool.filepath.is_empty() {
                        "file preview".to_owned()
                    } else if activity.tool.language == "diff" {
                        format!("edit diff: {}", activity.tool.filepath)
                    } else {
                        format!("file draft: {}", activity.tool.filepath)
                    };
                    tool_lines.push(Line::from(Span::styled(
                        format!("  · {preview_label}"),
                        Style::default().fg(app.theme.subtle),
                    )));
                    let language = if activity.tool.language.is_empty() {
                        language_for_filepath(&activity.tool.filepath)
                    } else {
                        activity.tool.language.as_str()
                    };
                    tool_lines.extend(preview_box_lines(
                        highlighted_code_lines(
                            &activity.tool.preview,
                            language,
                            &app.theme.syntax_theme,
                            app.theme.text,
                        ),
                        usize::from(content_width.saturating_sub(2)),
                        app.theme.border,
                        app.theme.panel,
                    ));
                    if activity.tool.preview_truncated {
                        tool_lines.push(Line::from(Span::styled(
                            "    … preview clipped; the complete content was still written",
                            Style::default()
                                .fg(app.theme.subtle)
                                .add_modifier(Modifier::DIM),
                        )));
                    }
                }
                push_spaced_message_block(&mut assistant_lines, tool_lines);
            }
        }
        if !turn.assistant.is_empty() {
            push_spaced_message_block(
                &mut assistant_lines,
                markdown_lines(
                    &turn.assistant,
                    app.theme.text,
                    app.theme.accent,
                    &app.theme.syntax_theme,
                ),
            );
        }
        if turn.state == "cancelled" && !turn.assistant.trim_end().ends_with("[interrupted]") {
            assistant_lines.push(Line::from(Span::styled(
                "[interrupted]",
                Style::default().fg(app.theme.warning),
            )));
        } else if turn.state == "error" {
            assistant_lines.push(Line::from(Span::styled(
                "[failed]",
                Style::default().fg(app.theme.error),
            )));
        }
        if !assistant_lines.is_empty() {
            lines.extend(message_rail_lines(
                assistant_lines,
                content_width,
                app.theme.accent,
            ));
        }
        lines.push(Line::default());
    }
    for message in app.command_messages.iter().filter(|message| {
        command_display_position(message, app.transcript.len()) == app.transcript.len()
    }) {
        append_command_lines(&mut lines, message, app, content_width);
    }
    let title = " Alphanus ";
    let viewport_height = usize::from(area.height.saturating_sub(2));
    let transcript_text = Text::from(lines);
    let visual_line_count = Paragraph::new(transcript_text.clone())
        .wrap(Wrap { trim: false })
        .line_count(content_width);
    let max_scroll = visual_line_count
        .saturating_sub(viewport_height)
        .min(usize::from(u16::MAX)) as u16;
    app.transcript_max_scroll = max_scroll;
    app.transcript_scroll = resolved_transcript_scroll(
        app.transcript_scroll,
        max_scroll,
        app.transcript_auto_follow,
    );
    let paragraph = Paragraph::new(transcript_text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(app.theme.border_type())
                .border_style(Style::default().fg(border))
                .title(title)
                .style(app.theme.base()),
        )
        .wrap(Wrap { trim: false })
        .scroll((app.transcript_scroll, 0));
    frame.render_widget(paragraph, area);
    let mut scrollbar = transcript_scrollbar_state(
        app.transcript_scroll,
        app.transcript_max_scroll,
        viewport_height,
    );
    frame.render_stateful_widget(
        Scrollbar::new(ScrollbarOrientation::VerticalRight)
            .style(Style::default().fg(app.theme.border)),
        area.inner(ratatui::layout::Margin {
            vertical: 1,
            horizontal: 0,
        }),
        &mut scrollbar,
    );
}

fn draw_attachments(frame: &mut Frame, app: &App, area: Rect) {
    let mut spans = Vec::new();
    for (index, attachment) in app.pending_attachments.iter().take(3).enumerate() {
        if index > 0 {
            spans.push(Span::raw(" "));
        }
        spans.push(Span::styled(
            format!(" {}. {} ", index + 1, field(attachment, "name")),
            Style::default().fg(app.theme.text).bg(app.theme.selection),
        ));
    }
    if app.pending_attachments.len() > 3 {
        spans.push(Span::styled(
            format!(" +{} more ", app.pending_attachments.len() - 3),
            Style::default().fg(app.theme.muted).bg(app.theme.selection),
        ));
    }
    frame.render_widget(
        Paragraph::new(Line::from(spans)).style(Style::default().bg(app.theme.panel)),
        area,
    );
}

fn draw_input(frame: &mut Frame, app: &mut App, area: Rect) {
    let border = if app.focus == Focus::Input {
        app.theme.accent
    } else {
        app.theme.border
    };
    let style = if app.input.is_empty() {
        Style::default().fg(app.theme.subtle)
    } else {
        Style::default().fg(app.theme.text)
    };
    frame.render_widget(
        Block::default()
            .borders(Borders::ALL)
            .border_type(app.theme.border_type())
            .border_style(Style::default().fg(border))
            .style(app.theme.base()),
        area,
    );
    let inner = area.inner(ratatui::layout::Margin {
        vertical: 1,
        horizontal: 1,
    });
    let show_file_button = inner.width >= 46;
    let button_width = if show_file_button { 18 } else { 0 };
    let parts = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(8), Constraint::Length(button_width)])
        .split(inner);
    app.composer_area = Rect::new(
        area.x,
        area.y,
        parts[0].right().saturating_sub(area.x),
        area.height,
    );
    app.file_button_area = parts[1];
    let value = if app.input.is_empty() {
        "Type a message…".into()
    } else {
        composer_view(&app.input, app.cursor, app.composer_area).0
    };
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                "> ",
                Style::default()
                    .fg(app.theme.accent)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(value, style),
        ])),
        parts[0],
    );
    if show_file_button {
        frame.render_widget(
            Paragraph::new(Line::from(vec![Span::styled(
                "+ File (Ctrl+F)",
                Style::default()
                    .fg(app.theme.secondary)
                    .add_modifier(Modifier::BOLD),
            )]))
            .alignment(Alignment::Right),
            parts[1],
        );
    }
}

fn draw_status(frame: &mut Frame, app: &App, area: Rect) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Length(1)])
        .split(area);
    let top = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
        .split(rows[0]);
    let bottom = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(45), Constraint::Percentage(55)])
        .split(rows[1]);
    let model_name = if app.model_name.is_empty() {
        "—"
    } else {
        &app.model_name
    };
    let state = app.model_state.trim().to_lowercase();
    let state_color = match state.as_str() {
        "online" => app.theme.success,
        "offline" => app.theme.error,
        _ => app.theme.muted,
    };
    let mut model_line = vec![
        Span::styled(" model: ", Style::default().fg(app.theme.subtle)),
        Span::styled(
            ellipsis(model_name, 36),
            Style::default()
                .fg(app.theme.accent)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("  ·  thinking: ", Style::default().fg(app.theme.subtle)),
        Span::styled(
            if app.thinking { "auto" } else { "off" },
            Style::default().fg(app.theme.secondary),
        ),
    ];
    if app.collaboration_mode == "plan" {
        model_line.extend([
            Span::styled("  ·  mode: ", Style::default().fg(app.theme.subtle)),
            Span::styled("plan", Style::default().fg(app.theme.warning)),
        ]);
    }
    frame.render_widget(
        Paragraph::new(Line::from(model_line)).style(Style::default().bg(app.theme.panel)),
        top[0],
    );

    let notice = app
        .clipboard_notice
        .as_ref()
        .filter(|(_, at)| at.elapsed() < Duration::from_secs(3))
        .map(|(text, _)| text.as_str());
    let mut hint_spans = if let Some(notice) = notice {
        vec![Span::styled(
            notice.to_owned(),
            Style::default().fg(app.theme.secondary),
        )]
    } else if app.approval.is_some() {
        vec![
            Span::styled(
                "Y",
                Style::default()
                    .fg(app.theme.success)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" approve    ", Style::default().fg(app.theme.muted)),
            Span::styled(
                "N",
                Style::default()
                    .fg(app.theme.error)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" deny", Style::default().fg(app.theme.muted)),
        ]
    } else if app.streaming {
        vec![
            Span::styled("Esc", Style::default().fg(app.theme.warning)),
            Span::styled(" twice to stop", Style::default().fg(app.theme.muted)),
        ]
    } else {
        match app.focus {
            Focus::Input if area.width < 110 => vec![
                Span::styled("Esc", Style::default().fg(app.theme.secondary)),
                Span::styled(" clear    ", Style::default().fg(app.theme.muted)),
            ],
            Focus::Input => vec![
                Span::styled("Esc", Style::default().fg(app.theme.secondary)),
                Span::styled(" clear    ", Style::default().fg(app.theme.muted)),
                Span::styled("Tab", Style::default().fg(app.theme.secondary)),
                Span::styled(" panel", Style::default().fg(app.theme.muted)),
            ],
            Focus::Transcript => vec![
                Span::styled("PgUp/PgDn", Style::default().fg(app.theme.secondary)),
                Span::styled(" scroll    ", Style::default().fg(app.theme.muted)),
                Span::styled("Tab", Style::default().fg(app.theme.secondary)),
                Span::styled(" panel", Style::default().fg(app.theme.muted)),
            ],
            Focus::Tree => vec![
                Span::styled("↑↓", Style::default().fg(app.theme.secondary)),
                Span::styled(" navigate    ", Style::default().fg(app.theme.muted)),
                Span::styled("Enter", Style::default().fg(app.theme.secondary)),
                Span::styled(" open    ", Style::default().fg(app.theme.muted)),
                Span::styled("[ ]", Style::default().fg(app.theme.secondary)),
                Span::styled(" siblings", Style::default().fg(app.theme.muted)),
            ],
        }
    };
    if notice.is_none() && !matches!(app.status.as_str(), "" | "Ready") && !app.streaming {
        hint_spans.insert(0, Span::styled("    ", Style::default()));
        hint_spans.insert(
            0,
            Span::styled(
                ellipsis(&app.status, 28),
                Style::default().fg(app.theme.warning),
            ),
        );
    }
    hint_spans.push(Span::raw(" "));
    frame.render_widget(
        Paragraph::new(Line::from(hint_spans))
            .alignment(Alignment::Right)
            .style(Style::default().bg(app.theme.panel)),
        top[1],
    );

    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(" session: ", Style::default().fg(app.theme.subtle)),
            Span::styled(
                ellipsis(&app.session_title, 22),
                Style::default().fg(app.theme.text),
            ),
            Span::styled("  ·  branch: ", Style::default().fg(app.theme.subtle)),
            Span::styled(
                ellipsis(&app.branch_name, 16),
                Style::default().fg(app.theme.accent),
            ),
        ]))
        .style(Style::default().bg(app.theme.panel)),
        bottom[0],
    );
    let context = match (app.context_tokens, app.context_window) {
        (Some(tokens), Some(window)) if window > 0 => {
            format!(
                "{}%",
                ((tokens.saturating_mul(100) + window / 2) / window).min(999)
            )
        }
        _ => "—".into(),
    };
    let endpoint = short_endpoint(&app.model_endpoint);
    let normalized_state = if state.is_empty() { "unknown" } else { &state };
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                ellipsis(&endpoint, 28),
                Style::default().fg(app.theme.muted),
            ),
            Span::styled("    ", Style::default()),
            Span::styled("● ", Style::default().fg(state_color)),
            Span::styled(
                normalized_state.to_owned(),
                Style::default()
                    .fg(state_color)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("    ctx ", Style::default().fg(app.theme.subtle)),
            Span::styled(context, Style::default().fg(app.theme.accent)),
            Span::raw(" "),
        ]))
        .alignment(Alignment::Right)
        .style(Style::default().bg(app.theme.panel)),
        bottom[1],
    );
}

fn short_endpoint(endpoint: &str) -> String {
    let without_scheme = endpoint
        .strip_prefix("https://")
        .or_else(|| endpoint.strip_prefix("http://"))
        .unwrap_or(endpoint);
    without_scheme
        .split('/')
        .next()
        .unwrap_or(without_scheme)
        .to_owned()
}

fn draw_sidebar(frame: &mut Frame, app: &App, area: Rect) {
    let sections = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(area);
    let items = app
        .tree
        .iter()
        .enumerate()
        .map(|(index, turn)| {
            let marker = if index == app.tree_selected {
                "›"
            } else {
                " "
            };
            let branch = if turn.branch_root { "⎇ " } else { "" };
            ListItem::new(format!("{marker} {branch}{}", ellipsis(&turn.user, 30)))
        })
        .collect::<Vec<_>>();
    let mut state = ListState::default().with_selected(Some(app.tree_selected));
    frame.render_stateful_widget(
        List::new(items)
            .highlight_style(app.theme.selected())
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(app.theme.border_type())
                    .border_style(Style::default().fg(if app.focus == Focus::Tree {
                        app.theme.secondary
                    } else {
                        app.theme.border
                    }))
                    .title(" Conversation Tree ")
                    .style(app.theme.base()),
            ),
        sections[0],
        &mut state,
    );
    let inspector = app
        .tree
        .get(app.tree_selected)
        .map(|turn| {
            format!(
                "id: {}\nstate: {}\nbranch: {}\nlabel: {}",
                turn.id,
                turn.state,
                turn.branch_root,
                if turn.label.is_empty() {
                    "—"
                } else {
                    &turn.label
                }
            )
        })
        .unwrap_or_else(|| "No selected turn".into());
    frame.render_widget(
        Paragraph::new(inspector)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(app.theme.border_type())
                    .border_style(Style::default().fg(app.theme.border))
                    .title(" Inspector ")
                    .style(app.theme.base()),
            )
            .wrap(Wrap { trim: false }),
        sections[1],
    );
}

fn approval_detail_lines(approval: &Approval) -> Vec<Line<'static>> {
    let mut lines = vec![Line::from(Span::styled(
        "Approval required",
        Style::default().add_modifier(Modifier::BOLD),
    ))];
    if !approval.reason.is_empty() {
        lines.push(Line::from(approval.reason.clone()));
    }
    if !approval.target.is_empty() {
        lines.push(Line::from(vec![
            Span::styled("Target  ", Style::default().add_modifier(Modifier::DIM)),
            Span::raw(approval.target.clone()),
        ]));
    }
    if !approval.command.is_empty() {
        lines.push(Line::from(vec![
            Span::styled("Command ", Style::default().add_modifier(Modifier::DIM)),
            Span::raw(approval.command.clone()),
        ]));
    }
    if !approval.cwd.is_empty() {
        lines.push(Line::from(vec![
            Span::styled("In      ", Style::default().add_modifier(Modifier::DIM)),
            Span::raw(approval.cwd.clone()),
        ]));
    }
    lines
}

fn approval_height(width: u16, approval: &Approval) -> u16 {
    let detail_width = width.saturating_sub(4).max(1);
    let visual_lines = Paragraph::new(Text::from(approval_detail_lines(approval)))
        .wrap(Wrap { trim: false })
        .line_count(detail_width);
    visual_lines.saturating_add(2).clamp(4, 8) as u16
}

fn draw_approval(frame: &mut Frame, app: &mut App, approval: &Approval, area: Rect) {
    app.approval_area = area;
    let rail = Block::default()
        .borders(Borders::LEFT)
        .border_type(app.theme.border_type())
        .border_style(Style::default().fg(app.theme.warning))
        .style(Style::default().bg(app.theme.panel));
    frame.render_widget(rail, area);
    let inner = Rect::new(
        area.x.saturating_add(3),
        area.y,
        area.width.saturating_sub(4),
        area.height,
    );
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(2), Constraint::Length(1)])
        .split(inner);
    let mut details = approval_detail_lines(approval);
    if let Some(title) = details.first_mut() {
        for span in &mut title.spans {
            span.style = span.style.fg(app.theme.warning);
        }
    }
    for line in details.iter_mut().skip(1) {
        for span in &mut line.spans {
            if span.style.fg.is_none() {
                span.style = span.style.fg(app.theme.text);
            }
        }
    }
    frame.render_widget(
        Paragraph::new(details)
            .style(Style::default().bg(app.theme.panel))
            .wrap(Wrap { trim: false }),
        rows[0],
    );
    let approve_label = "[Y] Approve";
    let deny_label = "[N] Deny";
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                approve_label,
                Style::default()
                    .fg(app.theme.success)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("    "),
            Span::styled(
                deny_label,
                Style::default()
                    .fg(app.theme.error)
                    .add_modifier(Modifier::BOLD),
            ),
        ]))
        .style(Style::default().bg(app.theme.panel)),
        rows[1],
    );
    app.approval_approve_area = Rect::new(rows[1].x, rows[1].y, approve_label.len() as u16, 1);
    app.approval_deny_area = Rect::new(
        rows[1].x + approve_label.len() as u16 + 4,
        rows[1].y,
        deny_label.len() as u16,
        1,
    );
}

fn draw_sessions_popup(
    frame: &mut Frame,
    app: &mut App,
    query: &str,
    selected: usize,
    items: &[Value],
) {
    const MAX_VISIBLE_SESSIONS: usize = 10;
    let visible_rows = items.len().clamp(3, MAX_VISIBLE_SESSIONS);
    let area = centered(frame.area(), 72, visible_rows as u16 + 10);
    app.popup_area = area;
    let inner = area.inner(ratatui::layout::Margin {
        vertical: 1,
        horizontal: 2,
    });
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(3),
            Constraint::Length(1),
            Constraint::Min(3),
            Constraint::Length(2),
        ])
        .split(inner);

    frame.render_widget(Clear, area);
    frame.render_widget(
        Block::default()
            .borders(Borders::ALL)
            .border_type(app.theme.border_type())
            .border_style(Style::default().fg(app.theme.accent))
            .title(Span::styled(
                " Sessions ",
                Style::default()
                    .fg(app.theme.accent)
                    .add_modifier(Modifier::BOLD),
            ))
            .style(app.theme.base()),
        area,
    );

    let button_width = rows[0].width.min(20);
    let heading_area = Rect::new(
        rows[0].x,
        rows[0].y,
        rows[0].width.saturating_sub(button_width),
        1,
    );
    app.popup_new_area = Rect::new(
        rows[0].right().saturating_sub(button_width),
        rows[0].y,
        button_width,
        1,
    );
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                "Recent conversations",
                Style::default()
                    .fg(app.theme.text)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("  ·  {} saved", items.len()),
                Style::default().fg(app.theme.muted),
            ),
        ])),
        heading_area,
    );
    frame.render_widget(
        Paragraph::new(Span::styled(
            "[ Ctrl+N  New ]",
            Style::default()
                .fg(app.theme.success)
                .add_modifier(Modifier::BOLD),
        ))
        .alignment(Alignment::Right),
        app.popup_new_area,
    );

    let search_text = if query.is_empty() {
        Span::styled("Search by name…", Style::default().fg(app.theme.subtle))
    } else {
        Span::styled(query.to_owned(), Style::default().fg(app.theme.text))
    };
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("⌕ ", Style::default().fg(app.theme.secondary)),
            search_text,
        ]))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(app.theme.border_type())
                .border_style(Style::default().fg(app.theme.secondary))
                .style(Style::default().bg(app.theme.panel)),
        ),
        rows[2],
    );

    let start = selected.saturating_sub(MAX_VISIBLE_SESSIONS - 1);
    app.popup_list_offset = start;
    app.popup_list_area = rows[4];
    if items.is_empty() {
        frame.render_widget(
            Paragraph::new(Text::from(vec![
                Line::from(Span::styled(
                    "No matching sessions",
                    Style::default()
                        .fg(app.theme.muted)
                        .add_modifier(Modifier::ITALIC),
                )),
                Line::from(Span::styled(
                    "Create one with Ctrl+N",
                    Style::default().fg(app.theme.subtle),
                )),
            ])),
            rows[4],
        );
    } else {
        let session_rows = items
            .iter()
            .enumerate()
            .skip(start)
            .take(MAX_VISIBLE_SESSIONS)
            .map(|(index, value)| {
                let is_selected = index == selected;
                let title = field(value, "title");
                let preview = field(value, "preview");
                let label = if preview.is_empty() {
                    title
                } else {
                    format!("{} · {}: {}", title, field(value, "kind"), preview)
                };
                let label = ellipsis(&label, 42);
                let turns = value.get("turn_count").and_then(Value::as_u64).unwrap_or(0);
                ListItem::new(Line::from(vec![
                    Span::styled(
                        if is_selected { "● " } else { "  " },
                        Style::default().fg(app.theme.accent),
                    ),
                    Span::styled(
                        format!("{label:<44}"),
                        Style::default()
                            .fg(if is_selected {
                                app.theme.text
                            } else {
                                app.theme.muted
                            })
                            .add_modifier(if is_selected {
                                Modifier::BOLD
                            } else {
                                Modifier::empty()
                            }),
                    ),
                    Span::styled(
                        format!("{turns} turns"),
                        Style::default().fg(if is_selected {
                            app.theme.secondary
                        } else {
                            app.theme.subtle
                        }),
                    ),
                ]))
                .style(if is_selected {
                    app.theme.selected()
                } else {
                    app.theme.base()
                })
            })
            .collect::<Vec<_>>();
        frame.render_widget(List::new(session_rows), rows[4]);
    }

    frame.render_widget(
        Paragraph::new(Text::from(vec![
            Line::from(vec![
                Span::styled("↑↓", Style::default().fg(app.theme.accent)),
                Span::styled(" Navigate    ", Style::default().fg(app.theme.muted)),
                Span::styled("Enter", Style::default().fg(app.theme.success)),
                Span::styled(" Open    ", Style::default().fg(app.theme.muted)),
                Span::styled("Delete ×2", Style::default().fg(app.theme.warning)),
                Span::styled(" Remove", Style::default().fg(app.theme.muted)),
            ]),
            Line::from(vec![
                Span::styled("Esc", Style::default().fg(app.theme.secondary)),
                Span::styled(" Close", Style::default().fg(app.theme.muted)),
            ]),
        ])),
        rows[5],
    );

    let query_width = UnicodeWidthStr::width(query) as u16;
    frame.set_cursor_position(Position::new(
        rows[2]
            .x
            .saturating_add(3)
            .saturating_add(query_width)
            .min(rows[2].right().saturating_sub(2)),
        rows[2].y + 1,
    ));
}

fn draw_popup(frame: &mut Frame, app: &mut App) {
    let Some(popup) = app.popup.clone() else {
        return;
    };
    if let Popup::Sessions {
        query,
        selected,
        items,
    } = &popup
    {
        draw_sessions_popup(frame, app, query, *selected, items);
        return;
    }
    app.popup_list_area = Rect::default();
    app.popup_new_area = Rect::default();
    app.popup_list_offset = 0;
    let (title, content, width, height) = match &popup {
        Popup::Palette {
            query,
            selected,
            mode,
        } => {
            let commands = filtered_palette(&app.command_catalog, query, *mode);
            let text = commands
                .iter()
                .enumerate()
                .take(12)
                .map(|(index, value)| {
                    format!(
                        "{} {:<28} {}",
                        if index == *selected { "›" } else { " " },
                        palette_prompt(value),
                        field(value, "description")
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
            (
                match mode {
                    PaletteMode::Commands => " Command Palette ",
                    PaletteMode::Global => " Global Palette ",
                    PaletteMode::Files => " File Picker ",
                },
                format!("Search: {query}\n\n{text}"),
                76,
                16,
            )
        }
        Popup::SessionName { value } => (
            " New Session ",
            format!("Name:\n\n{value}\n\n[Create]                         [Cancel]"),
            58,
            9,
        ),
        Popup::Theme { selected, items } => {
            let text = items
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    format!(
                        "{} {}",
                        if index == *selected { "›" } else { " " },
                        field(value, "title")
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
            (" Theme ", text, 54, 20)
        }
        Popup::Config { value, .. } => (
            " Configuration ",
            format!("Ctrl+S save · Ctrl+Y copy · Esc cancel\n\n{value}\n\n[Save]                         [Cancel]"),
            90,
            32,
        ),
        Popup::Health { report } => (" Health ", report.clone(), 84, 30),
        Popup::Code { content } => (
            " Code Viewer ",
            format!("Y copy · Esc close\n\n{content}"),
            90,
            30,
        ),
        Popup::Fatal => {
            let diagnostics = app
                .diagnostics
                .iter()
                .rev()
                .take(8)
                .rev()
                .cloned()
                .collect::<Vec<_>>()
                .join("\n");
            (
                " Runtime Failure ",
                format!("{}\n\n{}\n\n[Restart]                         [Quit]", app.status, diagnostics),
                82,
                18,
            )
        }
        Popup::Sessions { .. } => unreachable!("sessions are rendered separately"),
    };
    let area = centered(frame.area(), width, height);
    app.popup_area = area;
    frame.render_widget(Clear, area);
    frame.render_widget(
        Paragraph::new(content)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(app.theme.border_type())
                    .border_style(Style::default().fg(app.theme.accent))
                    .title(title)
                    .style(app.theme.base()),
            )
            .wrap(Wrap { trim: false }),
        area,
    );
}

fn markdown_lines(
    source: &str,
    text: ratatui::style::Color,
    accent: ratatui::style::Color,
    syntax_theme: &str,
) -> Vec<Line<'static>> {
    if source.is_empty() {
        return vec![Line::from(Span::styled("…", Style::default().fg(text)))];
    }
    let mut lines = vec![Line::default()];
    let mut bold = false;
    let mut code_language: Option<String> = None;
    for event in Parser::new(source) {
        match event {
            MarkdownEvent::Start(Tag::Strong) => bold = true,
            MarkdownEvent::End(TagEnd::Strong) => bold = false,
            MarkdownEvent::Start(Tag::CodeBlock(kind)) => {
                code_language = Some(match kind {
                    CodeBlockKind::Fenced(language) => language.into_string(),
                    CodeBlockKind::Indented => String::new(),
                });
                lines.push(Line::default());
            }
            MarkdownEvent::End(TagEnd::CodeBlock) => {
                code_language = None;
                lines.push(Line::default());
            }
            MarkdownEvent::Code(value) => {
                if let Some(line) = lines.last_mut() {
                    line.spans.push(Span::styled(
                        value.into_string(),
                        Style::default().fg(accent).bg(ratatui::style::Color::Black),
                    ));
                }
            }
            MarkdownEvent::Text(value) if code_language.is_some() => {
                lines.extend(highlighted_code_lines(
                    &value,
                    code_language.as_deref().unwrap_or_default(),
                    syntax_theme,
                    accent,
                ));
            }
            MarkdownEvent::Text(value) => {
                for (index, part) in value.split('\n').enumerate() {
                    if index > 0 {
                        lines.push(Line::default());
                    }
                    let mut style = Style::default().fg(text);
                    if bold {
                        style = style.add_modifier(Modifier::BOLD);
                    }
                    if let Some(line) = lines.last_mut() {
                        line.spans.push(Span::styled(part.to_string(), style));
                    }
                }
            }
            MarkdownEvent::SoftBreak | MarkdownEvent::HardBreak => lines.push(Line::default()),
            MarkdownEvent::Start(Tag::Item) => {
                if let Some(line) = lines.last_mut() {
                    line.spans
                        .push(Span::styled("• ", Style::default().fg(accent)));
                }
            }
            MarkdownEvent::End(TagEnd::Paragraph | TagEnd::Item | TagEnd::Heading(_)) => {
                lines.push(Line::default())
            }
            _ => {}
        }
    }
    lines
}

fn highlighted_code_lines(
    source: &str,
    language: &str,
    requested_theme: &str,
    fallback: ratatui::style::Color,
) -> Vec<Line<'static>> {
    static SYNTAXES: OnceLock<SyntaxSet> = OnceLock::new();
    static THEMES: OnceLock<ThemeSet> = OnceLock::new();
    let syntaxes = SYNTAXES.get_or_init(SyntaxSet::load_defaults_newlines);
    let themes = THEMES.get_or_init(ThemeSet::load_defaults);
    let syntax = syntaxes
        .find_syntax_by_token(language)
        .unwrap_or_else(|| syntaxes.find_syntax_plain_text());
    let theme = themes
        .themes
        .get(requested_theme)
        .or_else(|| themes.themes.get("base16-ocean.dark"));
    let Some(theme) = theme else {
        return source
            .lines()
            .map(|line| {
                Line::from(Span::styled(
                    line.to_owned(),
                    Style::default()
                        .fg(fallback)
                        .bg(ratatui::style::Color::Black),
                ))
            })
            .collect();
    };
    let mut highlighter = HighlightLines::new(syntax, theme);
    LinesWithEndings::from(source)
        .map(|line| match highlighter.highlight_line(line, syntaxes) {
            Ok(regions) => Line::from(
                regions
                    .into_iter()
                    .map(|(style, text)| {
                        Span::styled(
                            text.trim_end_matches(['\r', '\n']).to_owned(),
                            Style::default()
                                .fg(ratatui::style::Color::Rgb(
                                    style.foreground.r,
                                    style.foreground.g,
                                    style.foreground.b,
                                ))
                                .bg(ratatui::style::Color::Black),
                        )
                    })
                    .collect::<Vec<_>>(),
            ),
            Err(_) => Line::from(Span::styled(line.to_owned(), Style::default().fg(fallback))),
        })
        .collect()
}

fn centered(area: Rect, requested_width: u16, requested_height: u16) -> Rect {
    let horizontal_margin = if area.width > 8 { 4 } else { 0 };
    let vertical_margin = if area.height > 6 { 2 } else { 0 };
    let width = requested_width.min(area.width.saturating_sub(horizontal_margin));
    let height = requested_height.min(area.height.saturating_sub(vertical_margin));
    Rect::new(
        area.x + area.width.saturating_sub(width) / 2,
        area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    )
}

fn field(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned()
}

fn ellipsis(value: &str, max: usize) -> String {
    let one_line = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if one_line.chars().count() <= max {
        one_line
    } else {
        format!("{}…", one_line.chars().take(max).collect::<String>())
    }
}

fn pretty_json(value: &Value) -> String {
    serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
}

fn push_bounded<T>(queue: &mut VecDeque<T>, item: T, limit: usize) {
    if queue.len() >= limit {
        queue.pop_front();
    }
    queue.push_back(item);
}

fn trim_vec<T>(items: &mut Vec<T>, limit: usize) {
    if items.len() > limit {
        items.drain(..items.len() - limit);
    }
}

fn append_reasoning(turns: &mut [TurnView], turn_id: &str, reasoning: &str) {
    if reasoning.is_empty() {
        return;
    }
    if let Some(turn) = turns.iter_mut().find(|turn| turn.id == turn_id) {
        if let Some(activity) = turn
            .activity
            .last_mut()
            .filter(|item| item.kind == "reasoning")
        {
            append_bounded(&mut activity.text, reasoning, MAX_STREAM_CHARS);
        } else {
            if turn.activity.len() >= MAX_ACTIVITY_ITEMS {
                turn.activity.remove(0);
            }
            let mut text = String::new();
            append_bounded(&mut text, reasoning, MAX_STREAM_CHARS);
            turn.activity.push(ActivityItem::reasoning(text));
        }
    }
}

fn set_assistant(turns: &mut [TurnView], turn_id: &str, assistant: &str) {
    if let Some(turn) = turns.iter_mut().find(|turn| turn.id == turn_id) {
        turn.assistant.clear();
        append_bounded(&mut turn.assistant, assistant, MAX_STREAM_CHARS);
    }
}

#[derive(Debug, Clone, Default)]
struct ToolPreview {
    filepath: String,
    content: String,
    language: String,
    truncated: bool,
}

fn canonical_tool_name(name: &str) -> &str {
    name.rsplit([':', '.']).next().unwrap_or(name)
}

fn language_for_filepath(filepath: &str) -> &str {
    match filepath
        .rsplit('.')
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "bash" | "sh" | "zsh" => "bash",
        "c" | "h" => "c",
        "cc" | "cpp" | "cxx" | "hh" | "hpp" | "hxx" => "cpp",
        "css" => "css",
        "htm" | "html" => "html",
        "js" | "cjs" | "mjs" => "javascript",
        "json" => "json",
        "md" => "markdown",
        "py" => "python",
        "ts" => "typescript",
        "tsx" => "tsx",
        _ => "text",
    }
}

fn bounded_tool_preview(content: &str) -> (String, bool) {
    let mut output = String::new();
    let mut truncated = false;
    for (index, line) in content.lines().enumerate() {
        if index >= MAX_TOOL_PREVIEW_LINES {
            truncated = true;
            break;
        }
        if !output.is_empty() {
            output.push('\n');
        }
        for character in line.chars() {
            if output.len() + character.len_utf8() > MAX_TOOL_PREVIEW_CHARS {
                truncated = true;
                break;
            }
            output.push(character);
        }
        if truncated {
            break;
        }
    }
    (output, truncated)
}

fn tool_preview_from_request(name: &str, data: &Value) -> ToolPreview {
    if !matches!(canonical_tool_name(name), "create_file" | "edit_file") {
        return ToolPreview::default();
    }
    let Some(arguments) = data.get("arguments").and_then(Value::as_object) else {
        return ToolPreview::default();
    };
    let filepath = arguments
        .get("filepath")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let Some(content) = arguments.get("content").and_then(Value::as_str) else {
        return ToolPreview {
            filepath,
            ..ToolPreview::default()
        };
    };
    let (content, truncated) = bounded_tool_preview(content);
    ToolPreview {
        filepath,
        content,
        language: String::new(),
        truncated,
    }
}

fn tool_preview_from_result(name: &str, data: &Value) -> ToolPreview {
    let canonical = canonical_tool_name(name);
    if !matches!(canonical, "create_file" | "edit_file") {
        return ToolPreview::default();
    }
    let Some(result_data) = data
        .get("result")
        .and_then(|result| result.get("data"))
        .and_then(Value::as_object)
    else {
        return ToolPreview::default();
    };
    let filepath = result_data
        .get("filepath")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let (field_name, language) = if canonical == "edit_file" {
        ("diff", "diff")
    } else {
        ("content_preview", "")
    };
    let Some(content) = result_data.get(field_name).and_then(Value::as_str) else {
        return ToolPreview {
            filepath,
            ..ToolPreview::default()
        };
    };
    let (content, locally_truncated) = bounded_tool_preview(content);
    let result_truncated = result_data
        .get(if canonical == "edit_file" {
            "diff_truncated"
        } else {
            "content_preview_truncated"
        })
        .and_then(Value::as_bool)
        .unwrap_or(false);
    ToolPreview {
        filepath,
        content,
        language: language.into(),
        truncated: locally_truncated || result_truncated,
    }
}

fn merge_tool_preview(tool: &mut ToolActivity, preview: &ToolPreview) {
    if !preview.filepath.is_empty() {
        tool.filepath.clone_from(&preview.filepath);
    }
    if !preview.content.is_empty() {
        tool.preview.clone_from(&preview.content);
        tool.language.clone_from(&preview.language);
        tool.preview_truncated = preview.truncated;
    }
}

fn matches_tool(tool: &ToolActivity, id: &str, stream_id: &str, name: &str) -> bool {
    (!stream_id.is_empty() && tool.stream_id == stream_id)
        || (!id.is_empty() && tool.id == id)
        || (id.is_empty() && stream_id.is_empty() && !tool.completed && tool.name == name)
}

fn append_tool_activity(
    turns: &mut [TurnView],
    turn_id: &str,
    id: &str,
    stream_id: &str,
    name: &str,
    preview: ToolPreview,
) {
    let Some(turn) = turns.iter_mut().find(|turn| turn.id == turn_id) else {
        return;
    };
    let mut tool = ToolActivity {
        id: id.to_owned(),
        stream_id: stream_id.to_owned(),
        name: name.to_owned(),
        completed: false,
        failed: false,
        filepath: String::new(),
        preview: String::new(),
        language: String::new(),
        preview_truncated: false,
    };
    merge_tool_preview(&mut tool, &preview);
    if let Some(existing) = turn
        .activity
        .iter_mut()
        .rev()
        .find(|item| item.kind == "tool" && matches_tool(&item.tool, id, stream_id, name))
    {
        if !id.is_empty() {
            existing.tool.id = id.to_owned();
        }
        if !stream_id.is_empty() {
            existing.tool.stream_id = stream_id.to_owned();
        }
        existing.tool.name = name.to_owned();
        merge_tool_preview(&mut existing.tool, &preview);
    } else {
        if turn.activity.len() >= MAX_ACTIVITY_ITEMS {
            turn.activity.remove(0);
        }
        turn.activity.push(ActivityItem::tool(tool));
    }
}

fn complete_tool_activity(
    turns: &mut [TurnView],
    turn_id: &str,
    id: &str,
    name: &str,
    preview: ToolPreview,
    failed: bool,
) {
    let Some(turn) = turns.iter_mut().find(|turn| turn.id == turn_id) else {
        return;
    };
    if let Some(activity) = turn.activity.iter_mut().rev().find(|activity| {
        activity.kind == "tool"
            && !activity.tool.completed
            && ((!id.is_empty() && activity.tool.id == id)
                || (id.is_empty() && activity.tool.name == name))
    }) {
        activity.tool.completed = true;
        activity.tool.failed = failed;
        merge_tool_preview(&mut activity.tool, &preview);
    } else {
        if turn.activity.len() >= MAX_ACTIVITY_ITEMS {
            turn.activity.remove(0);
        }
        turn.activity.push(ActivityItem::tool(ToolActivity {
            id: id.to_owned(),
            stream_id: String::new(),
            name: name.to_owned(),
            completed: true,
            failed,
            filepath: preview.filepath,
            preview: preview.content,
            language: preview.language,
            preview_truncated: preview.truncated,
        }));
    }
}

fn extract_partial_json_string(raw: &str, key: &str) -> Option<String> {
    let marker = format!("\"{key}\"");
    let start = raw.find(&marker)? + marker.len();
    let after_colon = raw[start..].find(':')? + start + 1;
    let value = raw[after_colon..].trim_start();
    let chars = value.strip_prefix('"')?.chars().collect::<Vec<_>>();
    let mut output = String::new();
    let mut index = 0;
    while index < chars.len() {
        let character = chars[index];
        if character == '"' {
            return Some(output);
        }
        if character != '\\' {
            output.push(character);
            index += 1;
            continue;
        }
        index += 1;
        let escaped = *chars.get(index)?;
        match escaped {
            'n' => output.push('\n'),
            'r' => output.push('\r'),
            't' => output.push('\t'),
            'b' => output.push('\u{0008}'),
            'f' => output.push('\u{000c}'),
            '"' | '\\' | '/' => output.push(escaped),
            'u' => {
                let hex = chars.get(index + 1..index + 5)?.iter().collect::<String>();
                if let Ok(value) = u32::from_str_radix(&hex, 16) {
                    if let Some(decoded) = char::from_u32(value) {
                        output.push(decoded);
                    }
                }
                index += 4;
            }
            _ => output.push(escaped),
        }
        index += 1;
    }
    Some(output)
}

fn update_tool_preview_delta(turns: &mut [TurnView], turn_id: &str, data: &Value) {
    let name = field(data, "name");
    if !matches!(canonical_tool_name(&name), "create_file" | "edit_file") {
        return;
    }
    let raw = field(data, "raw_arguments");
    let Some(content) = extract_partial_json_string(&raw, "content") else {
        return;
    };
    let filepath = extract_partial_json_string(&raw, "filepath").unwrap_or_default();
    let (content, truncated) = bounded_tool_preview(&content);
    append_tool_activity(
        turns,
        turn_id,
        &field(data, "id"),
        &field(data, "stream_id"),
        &name,
        ToolPreview {
            filepath,
            content,
            language: String::new(),
            truncated,
        },
    );
}

fn set_activity(turns: &mut [TurnView], turn_id: &str, activity: Vec<ActivityItem>) {
    if let Some(turn) = turns.iter_mut().find(|turn| turn.id == turn_id) {
        turn.activity = activity;
    }
}

fn append_bounded(destination: &mut String, chunk: &str, limit: usize) {
    if destination.len() >= limit {
        return;
    }
    let remaining = limit - destination.len();
    if chunk.len() <= remaining {
        destination.push_str(chunk);
        return;
    }
    let boundary = chunk
        .char_indices()
        .map(|(index, _)| index)
        .take_while(|index| *index <= remaining)
        .last()
        .unwrap_or(0);
    destination.push_str(&chunk[..boundary]);
    destination.push_str("\n… [stream truncated]");
}

fn should_retain_draft(streaming: bool, value: &str) -> bool {
    streaming && !value.starts_with('/')
}

fn terminal_mouse_report_len(value: &str) -> Option<usize> {
    let prefix_len = if value.starts_with("\u{1b}[<") {
        3
    } else if value.starts_with("[<") {
        2
    } else {
        return None;
    };
    let bytes = value.as_bytes();
    for index in prefix_len..bytes.len() {
        if matches!(bytes[index], b'M' | b'm') {
            let fields = &value[prefix_len..index];
            if fields.split(';').count() == 3
                && fields.split(';').all(|field| {
                    !field.is_empty() && field.bytes().all(|byte| byte.is_ascii_digit())
                })
            {
                return Some(index + 1);
            }
            return None;
        }
        if !matches!(bytes[index], b'0'..=b'9' | b';') {
            return None;
        }
    }
    None
}

fn strip_terminal_mouse_reports(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    let mut offset = 0;
    while offset < value.len() {
        let remaining = &value[offset..];
        if let Some(length) = terminal_mouse_report_len(remaining) {
            offset += length;
            continue;
        }
        let character = remaining
            .chars()
            .next()
            .expect("remaining input is non-empty");
        output.push(character);
        offset += character.len_utf8();
    }
    output
}

fn palette_prompt(value: &Value) -> String {
    let prompt = field(value, "prompt");
    if prompt.is_empty() {
        field(value, "command")
    } else {
        prompt
    }
}

fn palette_value(value: &Value) -> String {
    let value_field = field(value, "value");
    if value_field.is_empty() {
        palette_prompt(value)
            .split_whitespace()
            .next()
            .unwrap_or_default()
            .into()
    } else {
        value_field
    }
}

fn palette_request(value: &Value) -> Option<(&'static str, Value)> {
    let selected = palette_value(value);
    match field(value, "kind").as_str() {
        "session" => Some(("session.load", json!({"session_id":selected}))),
        "file" => Some(("attachment.add", json!({"path":selected}))),
        "skill" => Some(("skill.toggle", json!({"skill_id":selected}))),
        _ => None,
    }
}

fn session_identity(value: &Value) -> &str {
    value
        .get("session_id")
        .and_then(Value::as_str)
        .filter(|item| !item.is_empty())
        .or_else(|| value.get("id").and_then(Value::as_str))
        .unwrap_or_default()
}

fn session_load_data(value: &Value) -> Value {
    let mut data = json!({"session_id":session_identity(value)});
    let turn_id = field(value, "turn_id");
    if !turn_id.is_empty() {
        data["turn_id"] = Value::String(turn_id);
    }
    data
}

fn filtered_palette(catalog: &[Value], query: &str, mode: PaletteMode) -> Vec<Value> {
    let needle = query.trim().to_lowercase();
    catalog
        .iter()
        .filter(|value| {
            let kind = field(value, "kind");
            let included = match mode {
                PaletteMode::Commands => kind == "command",
                PaletteMode::Files => kind == "file",
                PaletteMode::Global => true,
            };
            included
                && (needle.is_empty()
                    || palette_prompt(value).to_lowercase().contains(&needle)
                    || field(value, "description").to_lowercase().contains(&needle)
                    || kind.to_lowercase().contains(&needle))
        })
        .cloned()
        .collect()
}

fn catalog_text(rows: &[Value], label_field: &str) -> String {
    let mut output = String::new();
    let mut section = "";
    for row in rows {
        let next_section = row.get("section").and_then(Value::as_str).unwrap_or("");
        if next_section != section {
            if !output.is_empty() {
                output.push('\n');
            }
            section = next_section;
            output.push_str(section);
            output.push('\n');
        }
        output.push_str(&format!(
            "  {:<28} {}\n",
            field(row, label_field),
            field(row, "description")
        ));
    }
    output.trim_end().into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;

    fn command_fixture() -> Vec<Value> {
        vec![
            json!({"kind":"command","value":"/memory-stats","prompt":"/memory-stats","description":"Show memory statistics"}),
            json!({"kind":"command","value":"/sessions","prompt":"/sessions","description":"Open sessions"}),
        ]
    }

    #[test]
    fn compact_paste_expands_without_losing_content() {
        let marker = "[Pasted 200 chars]".to_string();
        let chunk = PasteChunk {
            marker: marker.clone(),
            text: "x".repeat(200),
        };
        assert_eq!(marker.replace(&chunk.marker, &chunk.text).len(), 200);
    }

    #[test]
    fn command_filter_matches_descriptions() {
        let catalog = command_fixture();
        assert_eq!(
            palette_value(&filtered_palette(&catalog, "memory", PaletteMode::Commands)[0]),
            "/memory-stats"
        );
    }

    #[test]
    fn slash_query_filters_inline_commands() {
        let catalog = command_fixture();
        let query = "/mem".strip_prefix('/').expect("slash prefix");
        let matches = filtered_palette(&catalog, query, PaletteMode::Commands);
        assert_eq!(palette_value(&matches[0]), "/memory-stats");
    }

    #[test]
    fn f1_help_is_a_keymap_not_duplicate_command_help() {
        let text = catalog_text(
            &[
                json!({"section":"INPUT","key":"Ctrl+Shift+K","description":"Delete to end of line"}),
            ],
            "key",
        );
        assert!(text.contains("Ctrl+Shift+K"));
        assert!(!text.contains("/sessions"));
    }

    #[test]
    fn active_turn_submission_keeps_the_composer_draft() {
        assert!(should_retain_draft(true, "next prompt"));
        assert!(!should_retain_draft(false, "next prompt"));
        assert!(!should_retain_draft(true, "/details"));
    }

    #[test]
    fn terminal_mouse_reports_are_not_submitted_as_prompt_text() {
        assert_eq!(
            strip_terminal_mouse_reports("\u{1b}[<0;86;47M/context"),
            "/context"
        );
        assert_eq!(
            strip_terminal_mouse_reports("[<0;86;47Mこんにちは[<35;90;50m世界"),
            "こんにちは世界"
        );
        assert_eq!(
            strip_terminal_mouse_reports("literal [<not-mouse"),
            "literal [<not-mouse"
        );
    }

    #[test]
    fn composer_cursor_tracks_the_character_after_input() {
        let area = Rect::new(10, 4, 40, 3);
        assert_eq!(composer_cursor_x("", 0, area), 13);
        assert_eq!(composer_cursor_x("word ", 5, area), 18);
        assert_eq!(composer_cursor_x("界", "界".len(), area), 15);
    }

    #[test]
    fn composer_scrolls_the_rendered_text_with_a_long_input() {
        let area = Rect::new(0, 0, 20, 3);
        let input = "abcdefghijklmnopqrstuvwxyz";
        let (visible, cursor_x) = composer_view(input, input.len(), area);

        assert!(!visible.starts_with('a'));
        assert!(visible.ends_with('z'));
        assert!(UnicodeWidthStr::width(visible.as_str()) <= 17);
        assert!(cursor_x < area.right());
    }

    #[test]
    fn question_mark_is_text_while_the_composer_is_focused() {
        let question = KeyEvent::new(KeyCode::Char('?'), KeyModifiers::SHIFT);
        let f1 = KeyEvent::new(KeyCode::F(1), KeyModifiers::NONE);
        assert!(!should_show_help(&question, Focus::Input));
        assert!(should_show_help(&question, Focus::Transcript));
        assert!(should_show_help(&f1, Focus::Input));
    }

    #[test]
    fn terminal_shortcuts_invalidate_integrated_terminal_back_buffers() {
        let ctrl_k = Event::Key(KeyEvent::new(KeyCode::Char('k'), KeyModifiers::CONTROL));
        let ctrl_l = Event::Key(KeyEvent::new(KeyCode::Char('l'), KeyModifiers::CONTROL));
        let ctrl_y = Event::Key(KeyEvent::new(KeyCode::Char('y'), KeyModifiers::CONTROL));
        let ctrl_p = Event::Key(KeyEvent::new(KeyCode::Char('p'), KeyModifiers::CONTROL));
        assert!(terminal_repaint_shortcut(&ctrl_k));
        assert!(terminal_repaint_shortcut(&ctrl_l));
        assert!(terminal_repaint_shortcut(&ctrl_y));
        assert!(!terminal_repaint_shortcut(&ctrl_p));
    }

    #[test]
    fn session_manager_is_only_automatic_on_initial_frontend_startup() {
        let mut initial_startup = true;
        let mut backend_restart = false;
        assert!(take_startup_session_manager(&mut initial_startup));
        assert!(!initial_startup);
        assert!(!take_startup_session_manager(&mut backend_restart));
    }

    #[test]
    fn touchpad_scroll_uses_natural_direction() {
        assert_eq!(touchpad_scroll(12, true, 20), 9);
        assert_eq!(touchpad_scroll(12, false, 20), 15);
        assert_eq!(touchpad_scroll(0, true, 20), 0);
        assert_eq!(touchpad_scroll(19, false, 20), 20);
    }

    #[test]
    fn transcript_scroll_follows_and_clamps_to_the_viewport() {
        assert_eq!(resolved_transcript_scroll(0, 42, true), 42);
        assert_eq!(resolved_transcript_scroll(18, 42, false), 18);
        assert_eq!(resolved_transcript_scroll(99, 42, false), 42);
    }

    #[test]
    fn transcript_scrollbar_thumb_reaches_the_bottom_track_cell() {
        let area = Rect::new(0, 0, 2, 12);
        let mut buffer = ratatui::buffer::Buffer::empty(area);
        let mut state = transcript_scrollbar_state(80, 80, 20);

        ratatui::widgets::StatefulWidget::render(
            Scrollbar::new(ScrollbarOrientation::VerticalRight),
            area,
            &mut buffer,
            &mut state,
        );

        // The final row is the end arrow, so the bottom of the thumb must
        // occupy the row immediately above it when the transcript is at EOF.
        assert_eq!(buffer[(area.right() - 1, area.bottom() - 2)].symbol(), "█");
    }

    #[test]
    fn message_rail_repeats_on_every_wrapped_display_line() {
        let lines = message_rail_lines(
            vec![Line::from(Span::styled(
                "abcdef",
                Style::default().fg(ratatui::style::Color::White),
            ))],
            5,
            ratatui::style::Color::Green,
        );

        assert_eq!(lines.len(), 2);
        assert!(lines.iter().all(|line| line.spans[0].content == "┃"));
        assert!(lines
            .iter()
            .all(|line| line.spans[0].style.fg == Some(ratatui::style::Color::Green)));
        assert_eq!(lines[0].spans[2].content, "abc");
        assert_eq!(lines[1].spans[2].content, "def");
    }

    #[test]
    fn assistant_blocks_get_one_dimmed_railed_spacer() {
        let mut blocks = Vec::new();
        push_spaced_message_block(
            &mut blocks,
            dim_message_lines(vec![Line::from(Span::raw("reasoning"))]),
        );
        push_spaced_message_block(&mut blocks, vec![Line::from(Span::raw("tool"))]);

        assert_eq!(blocks.len(), 3);
        assert!(line_is_empty(&blocks[1]));
        assert!(blocks[0].spans[0]
            .style
            .add_modifier
            .contains(Modifier::DIM));

        let railed = message_rail_lines(blocks, 40, ratatui::style::Color::Yellow);
        assert_eq!(railed.len(), 3);
        assert!(railed.iter().all(|line| line.spans[0].content == "┃"));
    }

    #[test]
    fn ratatui_word_wrapping_sets_the_real_scroll_extent() {
        let paragraph = Paragraph::new("aaaaaa aaaaaa aaaaaa").wrap(Wrap { trim: false });
        assert_eq!(paragraph.line_count(10), 3);
    }

    #[test]
    fn dialogs_use_fixed_sizes_and_cap_to_small_terminals() {
        assert_eq!(
            centered(Rect::new(0, 0, 200, 60), 72, 13),
            Rect::new(64, 23, 72, 13)
        );
        assert_eq!(
            centered(Rect::new(0, 0, 40, 10), 72, 13),
            Rect::new(2, 1, 36, 8)
        );
    }

    #[test]
    fn approval_strip_is_compact_and_omits_empty_metadata() {
        let approval = Approval {
            id: "approval-one".into(),
            reason: "External file write requires approval".into(),
            target: "/tmp/demo.py".into(),
            command: String::new(),
            cwd: String::new(),
        };
        let lines = approval_detail_lines(&approval);
        let text = lines
            .iter()
            .flat_map(|line| line.spans.iter())
            .map(|span| span.content.as_ref())
            .collect::<Vec<_>>()
            .join(" ");

        assert_eq!(approval_height(120, &approval), 5);
        assert!(text.contains("/tmp/demo.py"));
        assert!(!text.contains("Command"));
        assert!(!text.contains("In      "));
    }

    #[test]
    fn status_shortens_model_endpoint_to_its_host() {
        assert_eq!(
            short_endpoint("http://127.0.0.1:8080/v1/chat/completions"),
            "127.0.0.1:8080"
        );
        assert_eq!(
            short_endpoint("https://api.example.com/v1"),
            "api.example.com"
        );
    }

    #[test]
    fn session_search_selection_loads_the_matched_turn() {
        let item = json!({
            "id":"composite-result-id",
            "session_id":"session-1",
            "turn_id":"turn-4"
        });
        assert_eq!(
            session_load_data(&item),
            json!({"session_id":"session-1","turn_id":"turn-4"})
        );
    }

    #[test]
    fn reasoning_stays_attached_when_a_completed_turn_is_reloaded() {
        let mut streaming = vec![TurnView {
            id: "turn-1".into(),
            ..TurnView::default()
        }];
        append_reasoning(&mut streaming, "turn-1", "first ");
        append_reasoning(&mut streaming, "turn-1", "second");
        assert_eq!(streaming[0].activity[0].text, "first second");

        let mut reloaded = vec![TurnView {
            id: "turn-1".into(),
            assistant: "final answer".into(),
            ..TurnView::default()
        }];
        set_activity(&mut reloaded, "turn-1", streaming[0].activity.clone());
        assert_eq!(reloaded[0].activity[0].text, "first second");
        assert_eq!(reloaded[0].assistant, "final answer");
    }

    #[test]
    fn tool_completion_updates_one_inline_entry() {
        let mut turns = vec![TurnView {
            id: "turn-1".into(),
            ..TurnView::default()
        }];
        append_tool_activity(
            &mut turns,
            "turn-1",
            "call-1",
            "",
            "recall_memory",
            ToolPreview::default(),
        );
        complete_tool_activity(
            &mut turns,
            "turn-1",
            "call-1",
            "recall_memory",
            ToolPreview::default(),
            false,
        );

        assert_eq!(turns[0].activity.len(), 1);
        assert_eq!(turns[0].activity[0].tool.name, "recall_memory");
        assert!(turns[0].activity[0].tool.completed);
    }

    #[test]
    fn failed_tool_completion_is_retained_inline() {
        let mut turns = vec![TurnView {
            id: "turn-1".into(),
            ..TurnView::default()
        }];
        append_tool_activity(
            &mut turns,
            "turn-1",
            "call-1",
            "",
            "shell_command",
            ToolPreview::default(),
        );
        complete_tool_activity(
            &mut turns,
            "turn-1",
            "call-1",
            "shell_command",
            ToolPreview::default(),
            true,
        );

        assert!(turns[0].activity[0].tool.completed);
        assert!(turns[0].activity[0].tool.failed);
    }

    #[test]
    fn streamed_create_file_arguments_update_one_live_preview() {
        let mut turns = vec![TurnView {
            id: "turn-1".into(),
            ..TurnView::default()
        }];
        update_tool_preview_delta(
            &mut turns,
            "turn-1",
            &json!({
                "stream_id":"stream-one",
                "id":"call-one",
                "name":"create_file",
                "raw_arguments":"{\"filepath\":\"demo.py\",\"content\":\"print('hel"
            }),
        );
        update_tool_preview_delta(
            &mut turns,
            "turn-1",
            &json!({
                "stream_id":"stream-one",
                "id":"call-one",
                "name":"create_file",
                "raw_arguments":"{\"filepath\":\"demo.py\",\"content\":\"print('hello')\\n\"}"
            }),
        );

        assert_eq!(turns[0].activity.len(), 1);
        assert_eq!(turns[0].activity[0].tool.filepath, "demo.py");
        assert_eq!(turns[0].activity[0].tool.preview, "print('hello')");
        assert_eq!(turns[0].activity[0].tool.stream_id, "stream-one");
    }

    #[test]
    fn live_preview_box_uses_theme_border_and_background() {
        let border = ratatui::style::Color::Cyan;
        let background = ratatui::style::Color::Blue;
        let lines = preview_box_lines(
            vec![Line::from(Span::styled(
                "print('hello')",
                Style::default().fg(ratatui::style::Color::Green),
            ))],
            60,
            border,
            background,
        );

        assert!(lines.first().unwrap().spans[0].content.starts_with('┌'));
        assert!(lines.last().unwrap().spans[0].content.starts_with('└'));
        assert_eq!(lines[0].spans[0].style.fg, Some(border));
        assert_eq!(lines[0].spans[0].style.bg, Some(background));
        assert_eq!(
            lines[1].spans[1].style.fg,
            Some(ratatui::style::Color::Green)
        );
        assert_eq!(lines[1].spans[1].style.bg, Some(background));
    }

    #[test]
    fn activity_keeps_reasoning_and_tools_in_stream_order() {
        let mut turns = vec![TurnView {
            id: "turn-1".into(),
            ..TurnView::default()
        }];
        append_reasoning(&mut turns, "turn-1", "inspect");
        append_tool_activity(
            &mut turns,
            "turn-1",
            "one",
            "",
            "inspect_file",
            ToolPreview::default(),
        );
        complete_tool_activity(
            &mut turns,
            "turn-1",
            "one",
            "inspect_file",
            ToolPreview::default(),
            false,
        );
        append_reasoning(&mut turns, "turn-1", "edit");
        append_tool_activity(
            &mut turns,
            "turn-1",
            "two",
            "",
            "edit_file",
            ToolPreview::default(),
        );
        complete_tool_activity(
            &mut turns,
            "turn-1",
            "two",
            "edit_file",
            ToolPreview::default(),
            false,
        );
        append_reasoning(&mut turns, "turn-1", "verify");

        assert_eq!(
            turns[0]
                .activity
                .iter()
                .map(|item| (
                    item.kind.as_str(),
                    item.text.as_str(),
                    item.tool.name.as_str()
                ))
                .collect::<Vec<_>>(),
            vec![
                ("reasoning", "inspect", ""),
                ("tool", "", "inspect_file"),
                ("reasoning", "edit", ""),
                ("tool", "", "edit_file"),
                ("reasoning", "verify", ""),
            ]
        );
        assert!(turns[0]
            .activity
            .iter()
            .filter(|item| item.kind == "tool")
            .all(|item| item.tool.completed));
    }

    #[test]
    fn snapshot_turn_restores_ordered_activity() {
        let turn = TurnView::from_value(&json!({
            "id":"turn-1",
            "assistant":"complete response",
            "activity":[
                {"kind":"reasoning","text":"before"},
                {"kind":"tool","id":"call-1","name":"create_file","completed":true},
                {"kind":"reasoning","text":"after"}
            ]
        }));

        assert_eq!(turn.assistant, "complete response");
        assert_eq!(turn.activity.len(), 3);
        assert_eq!(turn.activity[2].text, "after");
    }

    #[test]
    fn command_output_keeps_its_chronological_turn_position() {
        let before_first_turn = CommandMessage {
            text: "Conversation cleared".into(),
            turn_position: 0,
        };
        let after_first_turn = CommandMessage {
            text: "Help".into(),
            turn_position: 1,
        };

        assert_eq!(command_display_position(&before_first_turn, 1), 0);
        assert_eq!(command_display_position(&after_first_turn, 1), 1);
        assert_eq!(command_display_position(&after_first_turn, 0), 0);

        let mut messages = VecDeque::from([after_first_turn]);
        shift_command_positions(&mut messages, 12);
        assert_eq!(messages[0].turn_position, 13);
    }

    #[test]
    fn transcript_is_bounded() {
        let mut items = (0..100_000).collect::<Vec<_>>();
        trim_vec(&mut items, MAX_TRANSCRIPT_ITEMS);
        assert_eq!(items.len(), MAX_TRANSCRIPT_ITEMS);
        assert_eq!(items[0], 100_000 - MAX_TRANSCRIPT_ITEMS);
    }

    fn fixture_snapshot(width: u16, height: u16) -> String {
        let backend = TestBackend::new(width, height);
        let mut terminal = Terminal::new(backend).expect("test terminal");
        let theme = Theme::default();
        terminal
            .draw(|frame| {
                let area = frame.area();
                let columns = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Min(30), Constraint::Length(if width >= 90 { 30 } else { 0 })])
                    .split(area);
                let rows = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Min(5), Constraint::Length(3), Constraint::Length(2)])
                    .split(columns[0]);
                frame.render_widget(
                    Paragraph::new("┃ Build a reliable terminal agent.\n\n┃ I’ll inspect the repository and stream progress here.\n┃ ▶ read_file\n┃ ✓ search_code")
                        .block(Block::default().borders(Borders::ALL).border_type(theme.border_type()).title(" Alphanus ").style(theme.base()))
                        .wrap(Wrap { trim: false }),
                    rows[0],
                );
                frame.render_widget(
                    Paragraph::new("> Type a message…")
                        .block(Block::default().borders(Borders::ALL).border_type(theme.border_type()).style(theme.base())),
                    rows[1],
                );
                frame.render_widget(Paragraph::new(" Ready\n Example · execute · fake-model · online").style(theme.base()), rows[2]);
                if columns[1].width > 0 {
                    frame.render_widget(
                        Paragraph::new("› first turn\n  ⎇ alternate path\n\nInspector\nstate: done")
                            .block(Block::default().borders(Borders::ALL).border_type(theme.border_type()).title(" Conversation Tree ").style(theme.base())),
                        columns[1],
                    );
                }
            })
            .expect("draw fixture");
        let buffer = terminal.backend().buffer();
        (0..height)
            .map(|y| {
                let mut line = String::new();
                for x in 0..width {
                    line.push_str(buffer[(x, y)].symbol());
                }
                line.trim_end().to_owned()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn snapshot_80x24() {
        insta::assert_snapshot!(fixture_snapshot(80, 24));
    }

    #[test]
    fn snapshot_120x40() {
        insta::assert_snapshot!(fixture_snapshot(120, 40));
    }

    #[test]
    fn snapshot_160x50() {
        insta::assert_snapshot!(fixture_snapshot(160, 50));
    }
}
