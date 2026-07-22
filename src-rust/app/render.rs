use super::*;

mod popups;
use popups::draw_popup;

pub(super) fn draw(frame: &mut Frame, app: &mut App) {
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
    app.areas.transcript = vertical[0];
    app.areas.input = vertical[3];
    app.areas.sidebar = sidebar;
    draw_transcript(frame, app, vertical[0]);
    draw_attachments(frame, app, vertical[1]);
    if let Some(approval) = app.approval.clone() {
        draw_approval(frame, app, &approval, vertical[2]);
    } else {
        app.areas.approval = Rect::default();
        app.areas.approval_approve = Rect::default();
        app.areas.approval_deny = Rect::default();
    }
    draw_input(frame, app, vertical[3]);
    draw_status(frame, app, vertical[4]);
    if sidebar.width > 0 {
        draw_sidebar(frame, app, sidebar);
    }
    if app.command_suggestions && app.popup.is_none() && app.approval.is_none() {
        draw_command_suggestions(frame, app);
    } else {
        app.areas.command_suggestions = Rect::default();
    }
    if app.popup.is_some() {
        draw_popup(frame, app);
    }
    if app.focus == Focus::Input && app.popup.is_none() && app.approval.is_none() {
        frame.set_cursor_position(Position::new(
            composer_cursor_x(&app.input, app.cursor, app.areas.composer),
            app.areas.input.y + 1,
        ));
    }
}

pub(super) fn draw_command_suggestions(frame: &mut Frame, app: &mut App) {
    let matches = app.command_matches();
    let visible_count = matches
        .len()
        .min(8)
        .min(app.areas.input.y.saturating_sub(2) as usize);
    if visible_count == 0 || app.areas.input.width < 12 {
        app.areas.command_suggestions = Rect::default();
        return;
    }
    let height = visible_count as u16 + 2;
    let x = app.areas.input.x.saturating_add(3);
    let available_width = app.areas.input.right().saturating_sub(x + 1);
    let width = available_width.min(60);
    if width < 12 {
        app.areas.command_suggestions = Rect::default();
        return;
    }
    let area = Rect::new(x, app.areas.input.y.saturating_sub(height), width, height);
    app.areas.command_suggestions = area;
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

pub(super) fn composer_view(input: &str, cursor: usize, area: Rect) -> (String, u16) {
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

pub(super) fn composer_cursor_x(input: &str, cursor: usize, area: Rect) -> u16 {
    composer_view(input, cursor, area).1
}

pub(super) fn should_show_help(key: &KeyEvent, focus: Focus) -> bool {
    matches!(key.code, KeyCode::F(1))
        || (focus != Focus::Input && matches!(key.code, KeyCode::Char('?')))
}

pub(super) fn touchpad_scroll(current: u16, scroll_up_event: bool, max_scroll: u16) -> u16 {
    if scroll_up_event {
        current.saturating_sub(3)
    } else {
        current.saturating_add(3).min(max_scroll)
    }
}

pub(super) fn resolved_transcript_scroll(current: u16, max_scroll: u16, auto_follow: bool) -> u16 {
    if auto_follow {
        max_scroll
    } else {
        current.min(max_scroll)
    }
}

pub(super) fn take_startup_session_manager(show_on_ready: &mut bool) -> bool {
    std::mem::take(show_on_ready)
}

pub(super) fn transcript_scrollbar_state(
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

pub(super) fn line_is_empty(line: &Line<'_>) -> bool {
    line.spans.iter().all(|span| span.content.trim().is_empty())
}

pub(super) fn trim_message_lines(mut lines: Vec<Line<'static>>) -> Vec<Line<'static>> {
    while lines.first().is_some_and(line_is_empty) {
        lines.remove(0);
    }
    while lines.last().is_some_and(line_is_empty) {
        lines.pop();
    }
    lines
}

pub(super) fn dim_message_lines(mut lines: Vec<Line<'static>>) -> Vec<Line<'static>> {
    for line in &mut lines {
        for span in &mut line.spans {
            span.style = span.style.add_modifier(Modifier::DIM);
        }
    }
    lines
}

pub(super) fn push_spaced_message_block(
    destination: &mut Vec<Line<'static>>,
    block: Vec<Line<'static>>,
) {
    let block = trim_message_lines(block);
    if block.is_empty() {
        return;
    }
    if !destination.is_empty() {
        destination.push(Line::default());
    }
    destination.extend(block);
}

pub(super) fn rail_line(color: ratatui::style::Color) -> Line<'static> {
    Line::from(vec![
        Span::styled("┃", Style::default().fg(color).add_modifier(Modifier::BOLD)),
        Span::raw(" "),
    ])
}

pub(super) fn push_styled_character(line: &mut Line<'static>, character: char, style: Style) {
    if let Some(last) = line.spans.last_mut().filter(|span| span.style == style) {
        last.content.to_mut().push(character);
    } else {
        line.spans.push(Span::styled(character.to_string(), style));
    }
}

pub(super) fn wrap_styled_characters(
    characters: &[(char, Style, usize)],
    available: usize,
    color: ratatui::style::Color,
) -> Vec<Line<'static>> {
    wrap_styled_characters_with_prefix(characters, available, &rail_line(color))
}

pub(super) fn wrap_styled_characters_with_prefix(
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

pub(super) fn wrap_styled_lines(
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

pub(super) fn preview_box_lines(
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

pub(super) fn message_rail_lines(
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

pub(super) fn append_notice_lines(
    lines: &mut Vec<Line<'static>>,
    text: &str,
    app: &App,
    content_width: u16,
) {
    let command_lines = trim_message_lines(markdown_lines(
        text,
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

pub(super) fn draw_transcript(frame: &mut Frame, app: &mut App, area: Rect) {
    let border = if app.focus == Focus::Transcript {
        app.theme.secondary
    } else {
        app.theme.border
    };
    let content_width = area.width.saturating_sub(2).max(1);
    let mut lines = Vec::<Line<'static>>::new();
    for turn in &app.transcript {
        if turn.local {
            append_notice_lines(&mut lines, &turn.assistant, app, content_width);
            continue;
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
            if let ActivityItem::Reasoning(text) = activity {
                if app.show_details && !text.trim().is_empty() {
                    push_spaced_message_block(
                        &mut assistant_lines,
                        dim_message_lines(markdown_lines(
                            text,
                            app.theme.muted,
                            app.theme.subtle,
                            &app.theme.syntax_theme,
                        )),
                    );
                }
            } else if let ActivityItem::Tool(tool) = activity {
                let mut tool_lines = Vec::new();
                let (marker, color) = if tool.failed {
                    ("✗", app.theme.error)
                } else if tool.completed {
                    ("✓", app.theme.success)
                } else {
                    ("▶", app.theme.warning)
                };
                tool_lines.push(Line::from(vec![
                    Span::styled(format!("{marker} "), Style::default().fg(color)),
                    Span::styled(tool.name.clone(), Style::default().fg(app.theme.muted)),
                ]));
                if app.show_details && !tool.preview.is_empty() {
                    let preview_label = if tool.filepath.is_empty() {
                        "file preview".to_owned()
                    } else if tool.language == "diff" {
                        format!("edit diff: {}", tool.filepath)
                    } else {
                        format!("file draft: {}", tool.filepath)
                    };
                    tool_lines.push(Line::from(Span::styled(
                        format!("  · {preview_label}"),
                        Style::default().fg(app.theme.subtle),
                    )));
                    let language = if tool.language.is_empty() {
                        tool_preview::language_for_filepath(&tool.filepath)
                    } else {
                        tool.language.as_str()
                    };
                    tool_lines.extend(preview_box_lines(
                        highlighted_code_lines(
                            &tool.preview,
                            language,
                            &app.theme.syntax_theme,
                            app.theme.text,
                        ),
                        usize::from(content_width.saturating_sub(2)),
                        app.theme.border,
                        app.theme.panel,
                    ));
                    if tool.preview_truncated {
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

pub(super) fn draw_attachments(frame: &mut Frame, app: &App, area: Rect) {
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

pub(super) fn draw_input(frame: &mut Frame, app: &mut App, area: Rect) {
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
    app.areas.composer = Rect::new(
        area.x,
        area.y,
        parts[0].right().saturating_sub(area.x),
        area.height,
    );
    app.areas.file_button = parts[1];
    let value = if app.input.is_empty() {
        "Type a message…".into()
    } else {
        composer_view(&app.input, app.cursor, app.areas.composer).0
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

pub(super) fn draw_status(frame: &mut Frame, app: &App, area: Rect) {
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
            Focus::Input if area.width < 110 => key_hints(&app.theme, &[("Esc", " clear    ")]),
            Focus::Input => key_hints(&app.theme, &[("Esc", " clear    "), ("Tab", " panel")]),
            Focus::Transcript => key_hints(
                &app.theme,
                &[("PgUp/PgDn", " scroll    "), ("Tab", " panel")],
            ),
            Focus::Tree => key_hints(
                &app.theme,
                &[
                    ("↑↓", " navigate    "),
                    ("Enter", " open    "),
                    ("[ ]", " siblings"),
                ],
            ),
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

pub(super) fn key_hints(theme: &Theme, items: &[(&str, &str)]) -> Vec<Span<'static>> {
    items
        .iter()
        .flat_map(|(key, action)| {
            [
                Span::styled((*key).to_owned(), Style::default().fg(theme.secondary)),
                Span::styled((*action).to_owned(), Style::default().fg(theme.muted)),
            ]
        })
        .collect()
}

pub(super) fn short_endpoint(endpoint: &str) -> String {
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

pub(super) fn draw_sidebar(frame: &mut Frame, app: &App, area: Rect) {
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

pub(super) fn approval_detail_lines(approval: &Approval) -> Vec<Line<'static>> {
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

pub(super) fn approval_height(width: u16, approval: &Approval) -> u16 {
    let detail_width = width.saturating_sub(4).max(1);
    let visual_lines = Paragraph::new(Text::from(approval_detail_lines(approval)))
        .wrap(Wrap { trim: false })
        .line_count(detail_width);
    visual_lines.saturating_add(2).clamp(4, 8) as u16
}

pub(super) fn draw_approval(frame: &mut Frame, app: &mut App, approval: &Approval, area: Rect) {
    app.areas.approval = area;
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
    app.areas.approval_approve = Rect::new(rows[1].x, rows[1].y, approve_label.len() as u16, 1);
    app.areas.approval_deny = Rect::new(
        rows[1].x + approve_label.len() as u16 + 4,
        rows[1].y,
        deny_label.len() as u16,
        1,
    );
}

pub(super) fn markdown_lines(
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

pub(super) fn highlighted_code_lines(
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

pub(super) fn centered(area: Rect, requested_width: u16, requested_height: u16) -> Rect {
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

pub(super) fn field(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned()
}

pub(super) fn values(value: &Value, key: &str) -> Vec<Value> {
    value
        .get(key)
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

pub(super) fn flag(value: &Value, key: &str) -> bool {
    value.get(key).and_then(Value::as_bool).unwrap_or(false)
}

pub(super) fn ellipsis(value: &str, max: usize) -> String {
    let one_line = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if one_line.chars().count() <= max {
        one_line
    } else {
        format!("{}…", one_line.chars().take(max).collect::<String>())
    }
}

pub(super) fn pretty_json(value: &Value) -> String {
    serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
}
