use super::*;

impl App {
    pub(super) fn new(
        python: &str,
        project_root: Option<&str>,
        debug: bool,
    ) -> Result<Self, String> {
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
            sidebar: false,
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
            areas: HitAreas::default(),
        })
    }

    pub(super) fn send(&mut self, kind: &str, data: Value) {
        if let Err(error) = self.backend.send(Request::new(kind, data)) {
            self.status = error;
        }
    }

    pub(super) fn restart_backend(&mut self) {
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

    pub(super) fn drain_backend(&mut self) {
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
                self.command_catalog = values(&frame.data, "commands");
                self.shortcut_catalog = values(&frame.data, "shortcuts");
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
                let preview = tool_preview::from_request(name, &frame.data);
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
                let preview = tool_preview::from_result(name, &frame.data);
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
                let items = values(&frame.data, "items");
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
                self.pending_attachments = values(&frame.data, "items");
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
                self.themes = values(&frame.data, "items");
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
                self.command_catalog = values(&frame.data, "items");
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
                combined.append(&mut self.transcript);
                self.transcript = combined;
                trim_vec(&mut self.transcript, MAX_TRANSCRIPT_ITEMS);
            } else {
                let mut combined = incoming;
                if !session_changed {
                    merge_local_notices(&mut combined, &self.transcript);
                }
                self.transcript = combined;
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
        let lines = values(value, "lines");
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
            self.append_notice(message);
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
}
