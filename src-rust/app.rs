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
use serde::Deserialize;
use serde_json::{json, Value};
use syntect::easy::HighlightLines;
use syntect::highlighting::ThemeSet;
use syntect::parsing::SyntaxSet;
use syntect::util::LinesWithEndings;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::backend::Backend;
use crate::input;
use crate::protocol::{BackendEvent, EventFrame, Request};
use crate::theme::Theme;
use crate::tool_preview::{self, ToolPreview};

mod interactions;
mod reducer;
mod render;
use render::*;

const MAX_EVENTS_PER_FRAME: usize = 256;
const MAX_TRANSCRIPT_ITEMS: usize = 4096;
const MAX_STREAM_CHARS: usize = 512 * 1024;
const MAX_ACTIVITY_ITEMS: usize = 256;
const MAX_DIAGNOSTICS: usize = 256;
const PASTE_THRESHOLD: usize = 120;
const CANCEL_WINDOW: Duration = Duration::from_secs(2);

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
    local: bool,
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
            branch_root: flag(value, "branch_root"),
            parent: field(value, "parent"),
            local: false,
        }
    }

    fn notice(text: String) -> Self {
        Self {
            assistant: text,
            local: true,
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize, PartialEq, Eq)]
#[serde(default)]
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

#[derive(Debug, Clone, PartialEq, Eq)]
enum ActivityItem {
    Reasoning(String),
    Tool(ToolActivity),
}

impl ActivityItem {
    fn reasoning(text: String) -> Self {
        Self::Reasoning(text)
    }

    fn tool(tool: ToolActivity) -> Self {
        Self::Tool(tool)
    }

    fn from_value(value: &Value) -> Option<Self> {
        match field(value, "kind").as_str() {
            "reasoning" => Some(Self::reasoning(field(value, "text"))),
            "tool" => serde_json::from_value(value.clone()).ok().map(Self::tool),
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
    areas: HitAreas,
}

#[derive(Default)]
struct HitAreas {
    transcript: Rect,
    sidebar: Rect,
    input: Rect,
    composer: Rect,
    file_button: Rect,
    popup: Rect,
    popup_list: Rect,
    popup_list_offset: usize,
    popup_new: Rect,
    approval: Rect,
    approval_approve: Rect,
    approval_deny: Rect,
    command_suggestions: Rect,
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

fn push_bounded<T>(queue: &mut VecDeque<T>, item: T, limit: usize) {
    if queue.len() >= limit {
        queue.pop_front();
    }
    queue.push_back(item);
}

fn trim_vec<T>(items: &mut Vec<T>, limit: usize) {
    items.drain(..items.len().saturating_sub(limit));
}

fn merge_local_notices(incoming: &mut Vec<TurnView>, existing: &[TurnView]) {
    let mut turn_position = 0;
    for notice in existing {
        if !notice.local {
            turn_position += 1;
            continue;
        }
        let index = incoming
            .iter()
            .enumerate()
            .filter(|(_, turn)| !turn.local)
            .nth(turn_position)
            .map_or(incoming.len(), |(index, _)| index);
        incoming.insert(index, notice.clone());
    }
}

fn append_reasoning(turns: &mut [TurnView], turn_id: &str, reasoning: &str) {
    if reasoning.is_empty() {
        return;
    }
    if let Some(turn) = turns.iter_mut().find(|turn| turn.id == turn_id) {
        if let Some(ActivityItem::Reasoning(text)) = turn.activity.last_mut() {
            append_bounded(text, reasoning, MAX_STREAM_CHARS);
        } else {
            let mut text = String::new();
            append_bounded(&mut text, reasoning, MAX_STREAM_CHARS);
            push_activity(turn, ActivityItem::reasoning(text));
        }
    }
}

fn push_activity(turn: &mut TurnView, activity: ActivityItem) {
    if turn.activity.len() >= MAX_ACTIVITY_ITEMS {
        turn.activity.remove(0);
    }
    turn.activity.push(activity);
}

fn set_assistant(turns: &mut [TurnView], turn_id: &str, assistant: &str) {
    if let Some(turn) = turns.iter_mut().find(|turn| turn.id == turn_id) {
        turn.assistant.clear();
        append_bounded(&mut turn.assistant, assistant, MAX_STREAM_CHARS);
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
        ..ToolActivity::default()
    };
    merge_tool_preview(&mut tool, &preview);
    if let Some(ActivityItem::Tool(existing)) = turn.activity.iter_mut().rev().find(
        |item| matches!(item, ActivityItem::Tool(tool) if matches_tool(tool, id, stream_id, name)),
    ) {
        if !id.is_empty() {
            existing.id = id.to_owned();
        }
        if !stream_id.is_empty() {
            existing.stream_id = stream_id.to_owned();
        }
        existing.name = name.to_owned();
        merge_tool_preview(existing, &preview);
    } else {
        push_activity(turn, ActivityItem::tool(tool));
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
    if let Some(ActivityItem::Tool(tool)) = turn.activity.iter_mut().rev().find(|activity| {
        matches!(activity, ActivityItem::Tool(tool) if !tool.completed
            && ((!id.is_empty() && tool.id == id) || (id.is_empty() && tool.name == name)))
    }) {
        tool.completed = true;
        tool.failed = failed;
        merge_tool_preview(tool, &preview);
    } else {
        push_activity(
            turn,
            ActivityItem::tool(ToolActivity {
                id: id.to_owned(),
                stream_id: String::new(),
                name: name.to_owned(),
                completed: true,
                failed,
                filepath: preview.filepath,
                preview: preview.content,
                language: preview.language,
                preview_truncated: preview.truncated,
            }),
        );
    }
}

fn update_tool_preview_delta(turns: &mut [TurnView], turn_id: &str, data: &Value) {
    let name = field(data, "name");
    if !matches!(
        tool_preview::canonical_name(&name),
        "create_file" | "edit_file"
    ) {
        return;
    }
    let raw = field(data, "raw_arguments");
    let Some(content) = tool_preview::partial_json_string(&raw, "content") else {
        return;
    };
    let filepath = tool_preview::partial_json_string(&raw, "filepath").unwrap_or_default();
    let (content, truncated) = tool_preview::bounded(&content);
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
    let mut boundary = remaining.min(chunk.len());
    while !chunk.is_char_boundary(boundary) {
        boundary -= 1;
    }
    destination.push_str(&chunk[..boundary]);
    destination.push_str("\n… [stream truncated]");
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
    let selected = field(value, "value");
    if selected.is_empty() {
        palette_prompt(value)
            .split_whitespace()
            .next()
            .unwrap_or_default()
            .into()
    } else {
        selected
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

fn session_identity(value: &Value) -> String {
    let identity = field(value, "session_id");
    if identity.is_empty() {
        field(value, "id")
    } else {
        identity
    }
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

    fn reasoning(item: &ActivityItem) -> &str {
        let ActivityItem::Reasoning(text) = item else {
            panic!("expected reasoning activity")
        };
        text
    }

    fn tool(item: &ActivityItem) -> &ToolActivity {
        let ActivityItem::Tool(tool) = item else {
            panic!("expected tool activity")
        };
        tool
    }

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
        let retains = |streaming: bool, value: &str| streaming && !value.starts_with('/');
        assert!(retains(true, "next prompt"));
        assert!(!retains(false, "next prompt"));
        assert!(!retains(true, "/details"));
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
        assert_eq!(reasoning(&streaming[0].activity[0]), "first second");

        let mut reloaded = vec![TurnView {
            id: "turn-1".into(),
            assistant: "final answer".into(),
            ..TurnView::default()
        }];
        set_activity(&mut reloaded, "turn-1", streaming[0].activity.clone());
        assert_eq!(reasoning(&reloaded[0].activity[0]), "first second");
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
        assert_eq!(tool(&turns[0].activity[0]).name, "recall_memory");
        assert!(tool(&turns[0].activity[0]).completed);
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

        assert!(tool(&turns[0].activity[0]).completed);
        assert!(tool(&turns[0].activity[0]).failed);
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
        assert_eq!(tool(&turns[0].activity[0]).filepath, "demo.py");
        assert_eq!(tool(&turns[0].activity[0]).preview, "print('hello')");
        assert_eq!(tool(&turns[0].activity[0]).stream_id, "stream-one");
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
                .map(|item| match item {
                    ActivityItem::Reasoning(text) => ("reasoning", text.as_str(), ""),
                    ActivityItem::Tool(tool) => ("tool", "", tool.name.as_str()),
                })
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
            .filter_map(|item| match item {
                ActivityItem::Tool(tool) => Some(tool),
                ActivityItem::Reasoning(_) => None,
            })
            .all(|tool| tool.completed));
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
        assert_eq!(reasoning(&turn.activity[2]), "after");
    }

    #[test]
    fn command_output_keeps_its_chronological_turn_position() {
        let turn = |id: &str| TurnView {
            id: id.into(),
            ..TurnView::default()
        };
        let existing = vec![turn("one"), TurnView::notice("Help".into()), turn("two")];
        let mut incoming = vec![turn("one"), turn("two")];

        merge_local_notices(&mut incoming, &existing);

        assert_eq!(incoming[1].assistant, "Help");
        assert!(incoming[1].local);
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
