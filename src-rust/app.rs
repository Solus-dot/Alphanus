use std::collections::VecDeque;
use std::io::{stdout, Write};
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
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use pulldown_cmark::{CodeBlockKind, Event as MarkdownEvent, Parser, Tag, TagEnd};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Position, Rect};
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
use unicode_width::UnicodeWidthStr;

use crate::backend::Backend;
use crate::protocol::{BackendEvent, EventFrame, Request};
use crate::theme::Theme;

const MAX_EVENTS_PER_FRAME: usize = 256;
const MAX_TRANSCRIPT_ITEMS: usize = 4096;
const MAX_STREAM_CHARS: usize = 512 * 1024;
const MAX_DIAGNOSTICS: usize = 256;
const PASTE_THRESHOLD: usize = 120;
const CANCEL_WINDOW: Duration = Duration::from_secs(2);

#[derive(Debug, Clone, Default)]
struct TurnView {
    id: String,
    user: String,
    attachments: String,
    assistant: String,
    state: String,
    label: String,
    branch_root: bool,
    parent: String,
}

impl TurnView {
    fn from_value(value: &Value) -> Self {
        Self {
            id: field(value, "id"),
            user: field(value, "user"),
            attachments: field(value, "attachments"),
            assistant: field(value, "assistant"),
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

#[derive(Debug, Clone)]
struct Approval {
    id: String,
    reason: String,
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
    Help,
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
    focus: Focus,
    input: String,
    cursor: usize,
    paste_chunks: Vec<PasteChunk>,
    transcript: Vec<TurnView>,
    tree: Vec<TurnView>,
    transcript_scroll: u16,
    tree_selected: usize,
    session_title: String,
    session_id: String,
    collaboration_mode: String,
    model_name: String,
    model_state: String,
    status: String,
    activity: VecDeque<String>,
    diagnostics: VecDeque<String>,
    pending_attachments: Vec<Value>,
    approval: Option<Approval>,
    popup: Option<Popup>,
    theme: Theme,
    themes: Vec<Value>,
    command_catalog: Vec<Value>,
    last_sequence: u64,
    last_escape: Option<Instant>,
    last_frame: Instant,
    active_assistant: String,
    active_reasoning: String,
    active_turn_id: String,
    clipboard_notice: Option<(String, Instant)>,
    session_delete_armed: Option<String>,
    transcript_offset: Option<usize>,
    transcript_previous: Option<usize>,
    tree_offset: Option<usize>,
    tree_previous: Option<usize>,
    transcript_area: Rect,
    sidebar_area: Rect,
    input_area: Rect,
    popup_area: Rect,
    approval_area: Rect,
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
            sidebar: true,
            focus: Focus::Input,
            input: String::new(),
            cursor: 0,
            paste_chunks: Vec::new(),
            transcript: Vec::new(),
            tree: Vec::new(),
            transcript_scroll: 0,
            tree_selected: 0,
            session_title: "Starting…".into(),
            session_id: String::new(),
            collaboration_mode: "execute".into(),
            model_name: String::new(),
            model_state: "connecting".into(),
            status: "Connecting to Python runtime…".into(),
            activity: VecDeque::new(),
            diagnostics: VecDeque::new(),
            pending_attachments: Vec::new(),
            approval: None,
            popup: None,
            theme: Theme::default(),
            themes: Vec::new(),
            command_catalog: default_commands(),
            last_sequence: 0,
            last_escape: None,
            last_frame: Instant::now(),
            active_assistant: String::new(),
            active_reasoning: String::new(),
            active_turn_id: String::new(),
            clipboard_notice: None,
            session_delete_armed: None,
            transcript_offset: None,
            transcript_previous: None,
            tree_offset: None,
            tree_previous: None,
            transcript_area: Rect::default(),
            sidebar_area: Rect::default(),
            input_area: Rect::default(),
            popup_area: Rect::default(),
            approval_area: Rect::default(),
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
                if let Some(snapshot) = frame.data.get("snapshot") {
                    self.apply_snapshot(snapshot);
                }
                self.send("theme.list", json!({}));
                self.send("palette.get", json!({}));
            }
            "state.snapshot" => self.apply_snapshot(&frame.data),
            "turn.started" => {
                self.streaming = true;
                self.active_turn_id = frame.turn_id.unwrap_or_default();
                self.active_assistant.clear();
                self.active_reasoning.clear();
                if let Some(turn) = frame.data.get("turn") {
                    self.transcript.push(TurnView::from_value(turn));
                    trim_vec(&mut self.transcript, MAX_TRANSCRIPT_ITEMS);
                }
                self.status = "Thinking…".into();
            }
            "assistant.delta" => {
                append_bounded(
                    &mut self.active_assistant,
                    frame.data.get("text").and_then(Value::as_str).unwrap_or(""),
                    MAX_STREAM_CHARS,
                );
                if let Some(turn) = self
                    .transcript
                    .iter_mut()
                    .find(|turn| turn.id == self.active_turn_id)
                {
                    turn.assistant = self.active_assistant.clone();
                }
            }
            "reasoning.delta" => {
                append_bounded(
                    &mut self.active_reasoning,
                    frame.data.get("text").and_then(Value::as_str).unwrap_or(""),
                    MAX_STREAM_CHARS,
                );
            }
            "tool.requested" => {
                let name = frame
                    .data
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("tool");
                push_bounded(&mut self.activity, format!("▶ {name}"), 128);
                self.status = format!("Running {name}…");
            }
            "tool.completed" => {
                let name = frame
                    .data
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("tool");
                push_bounded(&mut self.activity, format!("✓ {name}"), 128);
                self.status = "Working…".into();
            }
            "approval.requested" => {
                let request = frame.data.get("request").unwrap_or(&Value::Null);
                self.approval = Some(Approval {
                    id: frame.approval_id.unwrap_or_default(),
                    reason: field(request, "reason"),
                    command: field(request, "command"),
                    cwd: field(request, "cwd"),
                });
                self.status = "Action waiting for approval · Y/N".into();
            }
            "turn.completed" => {
                self.streaming = false;
                self.approval = None;
                self.active_reasoning.clear();
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
                self.model_name = field(&frame.data, "model_name");
            }
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
                    .unwrap_or_else(default_commands);
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
        if let Some(session) = value.get("session") {
            self.session_id = field(session, "id");
            self.session_title = field(session, "title");
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
                combined.append(&mut self.transcript);
                self.transcript = combined;
                trim_vec(&mut self.transcript, MAX_TRANSCRIPT_ITEMS);
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
            self.model_name = field(model, "model_name");
        }
        self.streaming = value
            .get("streaming")
            .and_then(Value::as_bool)
            .unwrap_or(self.streaming);
    }

    fn apply_command_result(&mut self, value: &Value) {
        let lines = value
            .get("lines")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        for line in lines.iter().filter_map(Value::as_str) {
            push_bounded(&mut self.activity, line.to_owned(), 128);
        }
        match value.get("action").and_then(Value::as_str).unwrap_or("") {
            "quit" => self.should_quit = true,
            "help" => self.popup = Some(Popup::Help),
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
            "toggle_details" => self.show_details = !self.show_details,
            "toggle_thinking" => self.thinking = !self.thinking,
            "code" => {
                self.popup = Some(Popup::Code {
                    content: self
                        .latest_code_block()
                        .unwrap_or_else(|| "No code blocks available".into()),
                })
            }
            _ => {}
        }
        if value.get("ok").and_then(Value::as_bool) == Some(false) {
            self.status = lines
                .first()
                .and_then(Value::as_str)
                .unwrap_or("Command failed")
                .into();
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
        if value.is_empty() || !self.connected {
            return;
        }
        self.input.clear();
        self.cursor = 0;
        self.paste_chunks.clear();
        if value.starts_with('/') {
            self.send("command.execute", json!({"command":value}));
        } else if !self.streaming {
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
                KeyCode::Char('b') => self.sidebar = !self.sidebar,
                KeyCode::Char('f') => {
                    self.popup = Some(Popup::Palette {
                        query: String::new(),
                        selected: 0,
                        mode: PaletteMode::Files,
                    })
                }
                KeyCode::Char('g') => self.focus = Focus::Input,
                KeyCode::Char('h') => self.focus = Focus::Transcript,
                KeyCode::Char('l') => self.focus = Focus::Tree,
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
        match key.code {
            KeyCode::F(1) | KeyCode::Char('?') => self.popup = Some(Popup::Help),
            KeyCode::F(2) => self.show_details = !self.show_details,
            KeyCode::F(3) => self.thinking = !self.thinking,
            KeyCode::Tab => {
                self.focus = match self.focus {
                    Focus::Input => Focus::Transcript,
                    Focus::Transcript => Focus::Tree,
                    Focus::Tree => Focus::Input,
                }
            }
            KeyCode::BackTab => {
                self.focus = match self.focus {
                    Focus::Input => Focus::Tree,
                    Focus::Tree => Focus::Transcript,
                    Focus::Transcript => Focus::Input,
                }
            }
            KeyCode::PageUp => {
                self.transcript_scroll = self.transcript_scroll.saturating_add(8);
                if let Some(previous) = self.transcript_previous.take() {
                    self.send(
                        "state.get",
                        json!({"transcript_offset":previous,"tree_offset":self.tree_offset}),
                    );
                }
            }
            KeyCode::PageDown => self.transcript_scroll = self.transcript_scroll.saturating_sub(8),
            KeyCode::Esc => self.handle_escape(),
            KeyCode::Char('/') if self.focus == Focus::Input && self.input.is_empty() => {
                self.popup = Some(Popup::Palette {
                    query: String::new(),
                    selected: 0,
                    mode: PaletteMode::Commands,
                });
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
            _ if self.focus == Focus::Input => self.edit_input(key),
            _ => {}
        }
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
            return;
        }
        let marker = format!("[Pasted {} chars]", text.chars().count());
        self.input.insert_str(self.cursor, &marker);
        self.cursor += marker.len();
        self.paste_chunks.push(PasteChunk { marker, text });
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
                    if let Some(id) = items
                        .get(*selected)
                        .and_then(|value| value.get("id"))
                        .and_then(Value::as_str)
                    {
                        deferred = Some(("session.load", json!({"session_id":id})));
                    }
                    self.popup = None;
                }
                KeyCode::Delete => {
                    if let Some(id) = items
                        .get(*selected)
                        .and_then(|value| value.get("id"))
                        .and_then(Value::as_str)
                    {
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
            Some(Popup::Help | Popup::Health { .. }) | None => {}
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
            if matches!(mouse.kind, MouseEventKind::Down(MouseButton::Left))
                && self
                    .approval_area
                    .contains(Position::new(mouse.column, mouse.row))
            {
                if let Some(approval) = self.approval.take() {
                    let approved =
                        mouse.column < self.approval_area.x + self.approval_area.width / 2;
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
        match mouse.kind {
            MouseEventKind::ScrollUp => {
                if self
                    .sidebar_area
                    .contains(Position::new(mouse.column, mouse.row))
                {
                    self.tree_selected = self.tree_selected.saturating_sub(1);
                } else {
                    self.transcript_scroll = self.transcript_scroll.saturating_add(3);
                }
            }
            MouseEventKind::ScrollDown => {
                if self
                    .sidebar_area
                    .contains(Position::new(mouse.column, mouse.row))
                {
                    self.tree_selected =
                        (self.tree_selected + 1).min(self.tree.len().saturating_sub(1));
                } else {
                    self.transcript_scroll = self.transcript_scroll.saturating_sub(3);
                }
            }
            MouseEventKind::Down(MouseButton::Left) => {
                let position = Position::new(mouse.column, mouse.row);
                if self.input_area.contains(position) {
                    self.focus = Focus::Input;
                } else if self.sidebar_area.contains(position) {
                    self.focus = Focus::Tree;
                    let relative = mouse.row.saturating_sub(self.sidebar_area.y + 3) as usize;
                    self.tree_selected = relative.min(self.tree.len().saturating_sub(1));
                } else if self.transcript_area.contains(position) {
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
                self.handle_popup_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
            }
            MouseEventKind::ScrollDown => {
                self.handle_popup_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE))
            }
            MouseEventKind::Down(MouseButton::Left) => {
                let relative_row = mouse.row.saturating_sub(self.popup_area.y) as usize;
                let relative_column = mouse.column.saturating_sub(self.popup_area.x);
                match self.popup.clone() {
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
                    Some(Popup::Sessions { items, .. }) if relative_row >= 4 => {
                        if let Some(id) = items
                            .get(relative_row - 4)
                            .and_then(|value| value.get("id"))
                            .and_then(Value::as_str)
                        {
                            let id = id.to_owned();
                            self.popup = None;
                            self.send("session.load", json!({"session_id":id}));
                        }
                    }
                    Some(Popup::Sessions { .. }) if relative_row == 2 => {
                        self.popup = Some(Popup::SessionName {
                            value: String::new(),
                        });
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
                    Some(Popup::Help | Popup::Health { .. } | Popup::Code { .. }) => {
                        self.popup = None
                    }
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
    let backend = CrosstermBackend::new(stdout());
    let mut terminal = Terminal::new(backend).map_err(|error| error.to_string())?;
    terminal.clear().map_err(|error| error.to_string())?;
    while !app.should_quit {
        app.drain_backend();
        if app.last_frame.elapsed() >= Duration::from_millis(16) {
            terminal
                .draw(|frame| draw(frame, &mut app))
                .map_err(|error| error.to_string())?;
            app.last_frame = Instant::now();
        }
        if event::poll(Duration::from_millis(8)).map_err(|error| error.to_string())? {
            let next = event::read().map_err(|error| error.to_string())?;
            app.handle_event(next);
        }
    }
    app.backend.shutdown();
    Ok(0)
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
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(5),
            Constraint::Length(1),
            Constraint::Length(3),
            Constraint::Length(2),
        ])
        .split(main);
    app.transcript_area = vertical[0];
    app.input_area = vertical[2];
    app.sidebar_area = sidebar;
    draw_transcript(frame, app, vertical[0]);
    draw_attachments(frame, app, vertical[1]);
    draw_input(frame, app, vertical[2]);
    draw_status(frame, app, vertical[3]);
    if sidebar.width > 0 {
        draw_sidebar(frame, app, sidebar);
    }
    if let Some(approval) = app.approval.clone() {
        app.approval_area = centered(frame.area(), 72, 15);
        draw_approval(frame, app, &approval, app.approval_area);
    }
    if app.popup.is_some() {
        draw_popup(frame, app);
    }
    if app.focus == Focus::Input && app.popup.is_none() && app.approval.is_none() {
        let prefix = 3usize;
        let width = app.input_area.width.saturating_sub(4) as usize;
        let before = &app.input[..app.cursor.min(app.input.len())];
        let cursor_width = UnicodeWidthStr::width(before);
        let visible = cursor_width.saturating_sub(width.saturating_sub(1));
        let x = app.input_area.x + 1 + (prefix + cursor_width.saturating_sub(visible)) as u16;
        frame.set_cursor_position(Position::new(
            x.min(app.input_area.right().saturating_sub(1)),
            app.input_area.y + 1,
        ));
    }
}

fn draw_transcript(frame: &mut Frame, app: &App, area: Rect) {
    let border = if app.focus == Focus::Transcript {
        app.theme.secondary
    } else {
        app.theme.border
    };
    let mut lines = Vec::<Line<'static>>::new();
    for turn in &app.transcript {
        lines.push(Line::from(Span::styled(
            "│ You",
            Style::default()
                .fg(app.theme.success)
                .add_modifier(Modifier::BOLD),
        )));
        if !turn.attachments.is_empty() {
            lines.push(Line::from(Span::styled(
                format!("│ Attachments: {}", turn.attachments),
                Style::default().fg(app.theme.muted),
            )));
        }
        lines.extend(markdown_lines(
            &turn.user,
            app.theme.text,
            app.theme.accent,
            &app.theme.syntax_theme,
        ));
        lines.push(Line::default());
        lines.push(Line::from(Span::styled(
            "│ Alphanus",
            Style::default()
                .fg(app.theme.accent)
                .add_modifier(Modifier::BOLD),
        )));
        lines.extend(markdown_lines(
            &turn.assistant,
            app.theme.text,
            app.theme.accent,
            &app.theme.syntax_theme,
        ));
        if turn.state == "cancelled" {
            lines.push(Line::from(Span::styled(
                "[interrupted]",
                Style::default().fg(app.theme.warning),
            )));
        } else if turn.state == "error" {
            lines.push(Line::from(Span::styled(
                "[failed]",
                Style::default().fg(app.theme.error),
            )));
        }
        lines.push(Line::default());
    }
    if app.streaming && !app.active_reasoning.is_empty() {
        lines.push(Line::from(Span::styled(
            "Reasoning",
            Style::default()
                .fg(app.theme.muted)
                .add_modifier(Modifier::ITALIC),
        )));
        lines.extend(markdown_lines(
            &app.active_reasoning,
            app.theme.muted,
            app.theme.subtle,
            &app.theme.syntax_theme,
        ));
    }
    if app.show_details && !app.activity.is_empty() {
        lines.push(Line::from(Span::styled(
            "Activity",
            Style::default()
                .fg(app.theme.muted)
                .add_modifier(Modifier::BOLD),
        )));
        lines.extend(app.activity.iter().rev().take(8).rev().map(|line| {
            Line::from(Span::styled(
                line.clone(),
                Style::default().fg(app.theme.subtle),
            ))
        }));
    }
    let title = if app.connected {
        format!(" Alphanus Alpha · {} ", app.session_title)
    } else {
        " Alphanus Alpha · connecting… ".into()
    };
    let paragraph = Paragraph::new(Text::from(lines))
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
    let mut scrollbar = ScrollbarState::new(app.transcript.len().saturating_mul(6))
        .position(app.transcript_scroll as usize);
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
    let text = if app.pending_attachments.is_empty() {
        String::new()
    } else {
        format!(
            " + {}",
            app.pending_attachments
                .iter()
                .filter_map(|value| value.get("name").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join(" · ")
        )
    };
    frame.render_widget(
        Paragraph::new(text).style(Style::default().fg(app.theme.muted).bg(app.theme.panel)),
        area,
    );
}

fn draw_input(frame: &mut Frame, app: &App, area: Rect) {
    let border = if app.focus == Focus::Input {
        app.theme.accent
    } else {
        app.theme.border
    };
    let value = if app.input.is_empty() {
        "Type a message…".into()
    } else {
        app.input.clone()
    };
    let style = if app.input.is_empty() {
        Style::default().fg(app.theme.subtle)
    } else {
        Style::default().fg(app.theme.text)
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
        ]))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(app.theme.border_type())
                .border_style(Style::default().fg(border))
                .style(app.theme.base()),
        ),
        area,
    );
}

fn draw_status(frame: &mut Frame, app: &App, area: Rect) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Length(1)])
        .split(area);
    let model = if app.model_name.is_empty() {
        app.model_state.clone()
    } else {
        format!("{} · {}", app.model_name, app.model_state)
    };
    let status = app
        .clipboard_notice
        .as_ref()
        .filter(|(_, at)| at.elapsed() < Duration::from_secs(3))
        .map(|(text, _)| text.as_str())
        .unwrap_or(&app.status);
    frame.render_widget(
        Paragraph::new(format!(" {status}"))
            .style(Style::default().fg(app.theme.muted).bg(app.theme.panel)),
        rows[0],
    );
    frame.render_widget(
        Paragraph::new(format!(
            " {} · {} · {} · F1 help · Ctrl+P commands",
            app.session_title, app.collaboration_mode, model
        ))
        .style(Style::default().fg(app.theme.subtle).bg(app.theme.panel)),
        rows[1],
    );
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

fn draw_approval(frame: &mut Frame, app: &App, approval: &Approval, area: Rect) {
    frame.render_widget(Clear, area);
    let text = format!(
        "Action Approval Required\n\nreason: {}\ncwd: {}\ncommand: {}\n\n[Y Approve]                         [N Deny]",
        approval.reason, approval.cwd, approval.command
    );
    frame.render_widget(
        Paragraph::new(text)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(app.theme.border_type())
                    .border_style(Style::default().fg(app.theme.warning))
                    .title(" Approval ")
                    .style(app.theme.base()),
            )
            .wrap(Wrap { trim: false }),
        area,
    );
}

fn draw_popup(frame: &mut Frame, app: &mut App) {
    let Some(popup) = &app.popup else {
        return;
    };
    let (title, content, width, height) = match popup {
        Popup::Help => (" Help ", help_text(), 78, 90),
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
                60,
            )
        }
        Popup::Sessions {
            query,
            selected,
            items,
        } => {
            let text = items
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    format!(
                        "{} {} · {} turns",
                        if index == *selected { "›" } else { " " },
                        field(value, "title"),
                        value.get("turn_count").and_then(Value::as_u64).unwrap_or(0)
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
            (
                " Sessions ",
                format!("Search: {query}\nCtrl+N new · Enter open · Delete twice remove · Esc close\n\n{text}"),
                72,
                75,
            )
        }
        Popup::SessionName { value } => (
            " New Session ",
            format!("Name:\n\n{value}\n\n[Create]                         [Cancel]"),
            58,
            28,
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
            (" Theme ", text, 54, 65)
        }
        Popup::Config { value, .. } => (
            " Configuration ",
            format!("Ctrl+S save · Ctrl+Y copy · Esc cancel\n\n{value}\n\n[Save]                         [Cancel]"),
            90,
            90,
        ),
        Popup::Health { report } => (" Health ", report.clone(), 84, 85),
        Popup::Code { content } => (
            " Code Viewer ",
            format!("Y copy · Esc close\n\n{content}"),
            90,
            85,
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
                55,
            )
        }
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

fn centered(area: Rect, width_percent: u16, height_percent: u16) -> Rect {
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - height_percent) / 2),
            Constraint::Percentage(height_percent),
            Constraint::Percentage((100 - height_percent) / 2),
        ])
        .split(area);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - width_percent) / 2),
            Constraint::Percentage(width_percent),
            Constraint::Percentage((100 - width_percent) / 2),
        ])
        .split(vertical[1])[1]
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

fn default_commands() -> Vec<Value> {
    let rows = [
        ("/help", "Show help"),
        ("/shortcuts", "Show keyboard shortcuts"),
        ("/details", "Toggle tool details"),
        ("/think", "Toggle thinking"),
        ("/mode", "Set plan or execute mode"),
        ("/clear", "Clear conversation"),
        ("/sessions", "Open sessions"),
        ("/rename", "Rename session"),
        ("/save", "Save session"),
        ("/file", "Attach a file"),
        ("/detach", "Remove attachments"),
        ("/branch", "Arm a branch"),
        ("/unbranch", "Leave a branch"),
        ("/branches", "List branches"),
        ("/switch", "Switch branch"),
        ("/tree", "Show conversation tree"),
        ("/skills", "List skills"),
        ("/reload", "Reload skills"),
        ("/doctor", "Run diagnostics"),
        ("/health", "Open health panel"),
        ("/skill-on", "Enable a skill"),
        ("/skill-off", "Disable a skill"),
        ("/skill-unload", "Unload a skill"),
        ("/skill-unload-all", "Unload all skills"),
        ("/skill-reload", "Reload skills"),
        ("/skill-info", "Show skill details"),
        ("/memory-stats", "Show memory stats"),
        ("/context", "Show context usage"),
        ("/audit", "Show file changes"),
        ("/project-tree", "Show project tree"),
        ("/theme", "Choose theme"),
        ("/config", "Edit config"),
        ("/report", "Save support report"),
        ("/code", "Open code viewer"),
        ("/quit", "Exit Alphanus"),
    ];
    rows.into_iter()
        .map(|(command, description)| {
            json!({"kind":"command","value":command,"prompt":command,"description":description})
        })
        .collect()
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

fn help_text() -> String {
    "CONVERSATION\n  /help /shortcuts /details /think /mode /clear\n  /sessions /rename /save /file /detach /quit\n\nBRANCHING\n  /branch /unbranch /branches /switch /tree\n\nSKILLS\n  /skills /reload /doctor /health /skill-on /skill-off\n  /skill-unload /skill-unload-all /skill-reload /skill-info\n\nUTILITIES\n  /memory-stats /context /audit /project-tree /theme /config /report /code\n\nKEYS\n  Ctrl+P commands · Ctrl+F file · Ctrl+B sidebar\n  F1 help · F2 details · F3 thinking · Esc twice cancel".into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;

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
        let catalog = default_commands();
        assert_eq!(
            palette_value(&filtered_palette(&catalog, "memory", PaletteMode::Commands)[0]),
            "/memory-stats"
        );
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
                    Paragraph::new("│ You\nBuild a reliable terminal agent.\n\n│ Alphanus\nI’ll inspect the repository and stream progress here.\n\nActivity\n▶ read_file\n✓ search_code")
                        .block(Block::default().borders(Borders::ALL).border_type(theme.border_type()).title(" Alphanus Alpha · Example ").style(theme.base()))
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
