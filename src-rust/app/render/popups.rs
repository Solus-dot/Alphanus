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

fn draw_session_name_popup(frame: &mut Frame, app: &mut App, value: &str) {
    let area = centered(frame.area(), 72, 11);
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
            Constraint::Min(1),
            Constraint::Length(1),
        ])
        .split(inner);

    frame.render_widget(Clear, area);
    frame.render_widget(
        Block::default()
            .borders(Borders::ALL)
            .border_type(app.theme.border_type())
            .border_style(Style::default().fg(app.theme.accent))
            .title(Span::styled(
                " New Session ",
                Style::default()
                    .fg(app.theme.accent)
                    .add_modifier(Modifier::BOLD),
            ))
            .style(app.theme.base()),
        area,
    );
    frame.render_widget(
        Paragraph::new(Span::styled(
            "Name your conversation",
            Style::default()
                .fg(app.theme.text)
                .add_modifier(Modifier::BOLD),
        )),
        rows[0],
    );
    let input = if value.is_empty() {
        Span::styled("Session name…", Style::default().fg(app.theme.subtle))
    } else {
        Span::styled(value.to_owned(), Style::default().fg(app.theme.text))
    };
    frame.render_widget(
        Paragraph::new(input).block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(app.theme.border_type())
                .border_style(Style::default().fg(app.theme.secondary))
                .style(Style::default().bg(app.theme.panel)),
        ),
        rows[2],
    );
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("Enter", Style::default().fg(app.theme.success)),
            Span::styled(" Create    ", Style::default().fg(app.theme.muted)),
            Span::styled("Esc", Style::default().fg(app.theme.secondary)),
            Span::styled(" Back", Style::default().fg(app.theme.muted)),
        ])),
        rows[4],
    );
    frame.set_cursor_position(Position::new(
        rows[2]
            .x
            .saturating_add(1)
            .saturating_add(UnicodeWidthStr::width(value) as u16)
            .min(rows[2].right().saturating_sub(2)),
        rows[2].y + 1,
    ));
}

fn draw_palette_popup(
    frame: &mut Frame,
    app: &mut App,
    query: &str,
    selected: usize,
    mode: PaletteMode,
) {
    const MAX_VISIBLE_ITEMS: usize = 10;
    let items = filtered_palette(palette_catalog(app, query, mode), query, mode);
    let loading = mode != PaletteMode::Commands && !app.palette_loaded;
    let area = centered(frame.area(), 76, MAX_VISIBLE_ITEMS as u16 + 10);
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
            Constraint::Length(1),
        ])
        .split(inner);
    let (title, heading, placeholder) = match mode {
        PaletteMode::Commands => (
            " Command Palette ",
            "Available commands",
            "Search commands…",
        ),
        PaletteMode::Global => (
            " Global Palette ",
            "Sessions, files, skills, and commands",
            "Search everything…",
        ),
        PaletteMode::Files => (
            " File Picker ",
            "Project files · paste any absolute path to attach",
            "Fuzzy-search file names or enter a path…",
        ),
    };

    frame.render_widget(Clear, area);
    frame.render_widget(
        Block::default()
            .borders(Borders::ALL)
            .border_type(app.theme.border_type())
            .border_style(Style::default().fg(app.theme.accent))
            .title(Span::styled(
                title,
                Style::default()
                    .fg(app.theme.accent)
                    .add_modifier(Modifier::BOLD),
            ))
            .style(app.theme.base()),
        area,
    );
    frame.render_widget(
        Paragraph::new(Span::styled(
            heading,
            Style::default()
                .fg(app.theme.text)
                .add_modifier(Modifier::BOLD),
        )),
        rows[0],
    );
    let search = if query.is_empty() {
        Span::styled(placeholder, Style::default().fg(app.theme.subtle))
    } else {
        Span::styled(query.to_owned(), Style::default().fg(app.theme.text))
    };
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("⌕ ", Style::default().fg(app.theme.secondary)),
            search,
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

    let start = selected.saturating_sub(MAX_VISIBLE_ITEMS - 1);
    app.areas.popup_list_offset = start;
    app.areas.popup_list = rows[4];
    if loading {
        frame.render_widget(
            Paragraph::new(Span::styled(
                "Loading…",
                Style::default()
                    .fg(app.theme.muted)
                    .add_modifier(Modifier::ITALIC),
            )),
            rows[4],
        );
    } else if items.is_empty() {
        frame.render_widget(
            Paragraph::new(Span::styled(
                "No matching items",
                Style::default()
                    .fg(app.theme.muted)
                    .add_modifier(Modifier::ITALIC),
            )),
            rows[4],
        );
    } else {
        let item_rows = items
            .iter()
            .enumerate()
            .skip(start)
            .take(MAX_VISIBLE_ITEMS)
            .map(|(index, value)| {
                let is_selected = index == selected;
                ListItem::new(Line::from(vec![
                    Span::styled(
                        if is_selected { "● " } else { "  " },
                        Style::default().fg(app.theme.accent),
                    ),
                    Span::styled(
                        format!("{:<30}", ellipsis(&palette_prompt(value), 28)),
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
                        ellipsis(&field(value, "description"), 36),
                        Style::default().fg(app.theme.subtle),
                    ),
                ]))
                .style(if is_selected {
                    app.theme.selected()
                } else {
                    app.theme.base()
                })
            })
            .collect::<Vec<_>>();
        frame.render_widget(List::new(item_rows), rows[4]);
    }
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("↑↓", Style::default().fg(app.theme.accent)),
            Span::styled(" Navigate    ", Style::default().fg(app.theme.muted)),
            Span::styled("Enter", Style::default().fg(app.theme.success)),
            Span::styled(
                if mode == PaletteMode::Files {
                    " Attach    "
                } else {
                    " Select    "
                },
                Style::default().fg(app.theme.muted),
            ),
            Span::styled(
                if mode == PaletteMode::Files {
                    "Tab"
                } else {
                    "Esc"
                },
                Style::default().fg(app.theme.secondary),
            ),
            Span::styled(
                if mode == PaletteMode::Files {
                    " Complete    Esc Close"
                } else {
                    " Close"
                },
                Style::default().fg(app.theme.muted),
            ),
        ])),
        rows[5],
    );
    frame.set_cursor_position(Position::new(
        rows[2]
            .x
            .saturating_add(3)
            .saturating_add(UnicodeWidthStr::width(query) as u16)
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
    if let Popup::SessionName { value } = &popup {
        draw_session_name_popup(frame, app, value);
        return;
    }
    if let Popup::Palette {
        query,
        selected,
        mode,
    } = &popup
    {
        draw_palette_popup(frame, app, query, *selected, *mode);
        return;
    }
    app.areas.popup_list = Rect::default();
    app.areas.popup_new = Rect::default();
    app.areas.popup_list_offset = 0;
    let (title, content, width, height) = match &popup {
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
        Popup::Sessions { .. } | Popup::SessionName { .. } | Popup::Palette { .. } => {
            unreachable!("custom popups are rendered separately")
        }
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
