use super::*;

impl App {
    pub(super) fn latest_code_block(&self) -> Option<String> {
        self.transcript
            .iter()
            .rev()
            .filter(|turn| !turn.local)
            .find_map(|turn| {
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
        if self.streaming && !value.starts_with('/') {
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

    pub(super) fn handle_event(&mut self, event: Event) {
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
        self.append_notice(catalog_text(&self.shortcut_catalog, "key"));
    }

    pub(super) fn append_notice(&mut self, text: String) {
        self.transcript.push(TurnView::notice(text));
        trim_vec(&mut self.transcript, MAX_TRANSCRIPT_ITEMS);
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

    pub(super) fn command_matches(&self) -> Vec<Value> {
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
        if !input::edit(&mut self.input, &mut self.cursor, key, false)
            && key.code == KeyCode::Backspace
            && self.input.is_empty()
        {
            self.send("attachment.remove", json!({"index":"last"}));
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
                        if self.session_delete_armed.as_deref() == Some(id.as_str()) {
                            deferred = Some(("session.delete", json!({"session_id":id})));
                            self.session_delete_armed = None;
                            self.popup = None;
                        } else {
                            self.session_delete_armed = Some(id);
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
                _ => {
                    input::edit(value, cursor, key, true);
                }
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
                let approved = self.areas.approval_approve.contains(position);
                let denied = self.areas.approval_deny.contains(position);
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
                .areas
                .command_suggestions
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
                        .saturating_sub(self.areas.command_suggestions.y + 1)
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
                    .areas
                    .sidebar
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
                    .areas
                    .sidebar
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
                if self.areas.file_button.contains(position) {
                    self.open_file_picker();
                } else if self.areas.input.contains(position) {
                    self.focus = Focus::Input;
                } else if self.areas.sidebar.contains(position) {
                    self.command_suggestions = false;
                    self.focus = Focus::Tree;
                    let relative = mouse.row.saturating_sub(self.areas.sidebar.y + 3) as usize;
                    self.tree_selected = relative.min(self.tree.len().saturating_sub(1));
                } else if self.areas.transcript.contains(position) {
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
        if !self.areas.popup.contains(position) {
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
                let relative_row = mouse.row.saturating_sub(self.areas.popup.y) as usize;
                let relative_column = mouse.column.saturating_sub(self.areas.popup.x);
                match self.popup.clone() {
                    Some(Popup::Sessions { .. }) if self.areas.popup_new.contains(position) => {
                        self.popup = Some(Popup::SessionName {
                            value: String::new(),
                        });
                    }
                    Some(Popup::Sessions { items, .. })
                        if self.areas.popup_list.contains(position) =>
                    {
                        let selected = self.areas.popup_list_offset
                            + mouse.row.saturating_sub(self.areas.popup_list.y) as usize;
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
                        if relative_row + 3 >= self.areas.popup.height as usize =>
                    {
                        if relative_column < self.areas.popup.width / 2 {
                            self.popup = None;
                            self.send("session.create", json!({"title":value}));
                        } else {
                            self.popup = None;
                        }
                    }
                    Some(Popup::Config { value, .. })
                        if relative_row + 3 >= self.areas.popup.height as usize =>
                    {
                        if relative_column < self.areas.popup.width / 2 {
                            self.popup = None;
                            self.send("config.apply", json!({"text":value}));
                        } else {
                            self.popup = None;
                        }
                    }
                    Some(Popup::Code { content }) if relative_row <= 2 => {
                        self.copy_to_clipboard(content)
                    }
                    Some(Popup::Fatal) if relative_row + 3 >= self.areas.popup.height as usize => {
                        if relative_column < self.areas.popup.width / 2 {
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
