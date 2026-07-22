use super::*;

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
    app.areas.popup = area;
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
    app.areas.popup_new = Rect::new(
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
        app.areas.popup_new,
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
    app.areas.popup_list_offset = start;
    app.areas.popup_list = rows[4];
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

pub(super) fn draw_popup(frame: &mut Frame, app: &mut App) {
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
    app.areas.popup_list = Rect::default();
    app.areas.popup_new = Rect::default();
    app.areas.popup_list_offset = 0;
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
    app.areas.popup = area;
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
