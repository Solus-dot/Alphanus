use super::*;

pub(in crate::app) fn line_is_empty(line: &Line<'_>) -> bool {
    line.spans.iter().all(|span| span.content.trim().is_empty())
}

pub(in crate::app) fn trim_message_lines(mut lines: Vec<Line<'static>>) -> Vec<Line<'static>> {
    while lines.first().is_some_and(line_is_empty) {
        lines.remove(0);
    }
    while lines.last().is_some_and(line_is_empty) {
        lines.pop();
    }
    lines
}

pub(in crate::app) fn dim_message_lines(mut lines: Vec<Line<'static>>) -> Vec<Line<'static>> {
    for line in &mut lines {
        for span in &mut line.spans {
            span.style = span.style.add_modifier(Modifier::DIM);
        }
    }
    lines
}

pub(in crate::app) fn push_spaced_message_block(
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

pub(in crate::app) fn rail_line(color: ratatui::style::Color) -> Line<'static> {
    Line::from(vec![
        Span::styled("┃", Style::default().fg(color).add_modifier(Modifier::BOLD)),
        Span::raw(" "),
    ])
}

pub(in crate::app) fn push_styled_character(
    line: &mut Line<'static>,
    character: char,
    style: Style,
) {
    if let Some(last) = line.spans.last_mut().filter(|span| span.style == style) {
        last.content.to_mut().push(character);
    } else {
        line.spans.push(Span::styled(character.to_string(), style));
    }
}

pub(in crate::app) fn wrap_styled_characters(
    characters: &[(char, Style, usize)],
    available: usize,
    color: ratatui::style::Color,
) -> Vec<Line<'static>> {
    wrap_styled_characters_with_prefix(characters, available, &rail_line(color))
}

pub(in crate::app) fn wrap_styled_characters_with_prefix(
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

pub(in crate::app) fn wrap_styled_lines(
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

pub(in crate::app) fn preview_box_lines(
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

pub(in crate::app) fn message_rail_lines(
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

pub(in crate::app) fn append_notice_lines(
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

pub(in crate::app) fn draw_transcript(frame: &mut Frame, app: &mut App, area: Rect) {
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
