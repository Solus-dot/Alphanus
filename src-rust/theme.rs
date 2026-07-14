use std::collections::HashMap;

use ratatui::style::{Color, Modifier, Style};
use ratatui::widgets::BorderType;
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct Theme {
    pub id: String,
    pub title: String,
    pub background: Color,
    pub panel: Color,
    pub text: Color,
    pub muted: Color,
    pub subtle: Color,
    pub accent: Color,
    pub secondary: Color,
    pub success: Color,
    pub warning: Color,
    pub error: Color,
    pub border: Color,
    pub selection: Color,
    pub border_set: String,
    pub syntax_theme: String,
    pub styles: HashMap<String, Style>,
}

impl Default for Theme {
    fn default() -> Self {
        Self {
            id: "catppuccin-mocha".into(),
            title: "Catppuccin Mocha".into(),
            background: rgb("#1e1e2e", Color::Black),
            panel: rgb("#181825", Color::Black),
            text: rgb("#cdd6f4", Color::White),
            muted: rgb("#a6adc8", Color::Gray),
            subtle: rgb("#7f849c", Color::DarkGray),
            accent: rgb("#cba6f7", Color::Magenta),
            secondary: rgb("#b4befe", Color::Blue),
            success: rgb("#a6e3a1", Color::Green),
            warning: rgb("#f9e2af", Color::Yellow),
            error: rgb("#f38ba8", Color::Red),
            border: rgb("#6c7086", Color::DarkGray),
            selection: rgb("#45475a", Color::DarkGray),
            border_set: "rounded".into(),
            syntax_theme: "base16-ocean.dark".into(),
            styles: HashMap::new(),
        }
    }
}

impl Theme {
    pub fn from_value(value: &Value) -> Self {
        let mut theme = Self::default();
        theme.id = text(value, "/id", &theme.id);
        theme.title = text(value, "/title", &theme.title);
        theme.background = color(value, "/theme/background", theme.background);
        theme.panel = color(
            value,
            "/colors/panel_bg",
            color(value, "/theme/panel", theme.panel),
        );
        theme.text = color(
            value,
            "/colors/text",
            color(value, "/theme/foreground", theme.text),
        );
        theme.muted = color(value, "/colors/muted", theme.muted);
        theme.subtle = color(value, "/colors/subtle", theme.subtle);
        theme.accent = color(value, "/colors/accent", theme.accent);
        theme.secondary = color(value, "/theme/secondary", theme.secondary);
        theme.success = color(value, "/colors/success", theme.success);
        theme.warning = color(value, "/colors/warning", theme.warning);
        theme.error = color(value, "/colors/error", theme.error);
        theme.border = color(value, "/colors/panel_border", theme.border);
        theme.selection = color(value, "/theme/variables/app-selection-bg", theme.selection);
        let border = value
            .pointer("/ratatui/border_set")
            .and_then(Value::as_str)
            .unwrap_or("rounded");
        if matches!(border, "plain" | "rounded" | "double" | "thick") {
            theme.border_set = border.into();
        }
        theme.syntax_theme = text(value, "/ratatui/syntax_theme", &theme.syntax_theme);
        for (key, destination) in [
            ("text", &mut theme.text),
            ("muted", &mut theme.muted),
            ("subtle", &mut theme.subtle),
            ("accent", &mut theme.accent),
            ("success", &mut theme.success),
            ("warning", &mut theme.warning),
            ("error", &mut theme.error),
            ("border", &mut theme.border),
            ("selection", &mut theme.selection),
        ] {
            let pointer = format!("/ratatui/styles/{key}/foreground");
            *destination = color(value, &pointer, *destination);
        }
        if let Some(overrides) = value.pointer("/ratatui/styles").and_then(Value::as_object) {
            for (key, raw) in overrides {
                let mut style = Style::default();
                if let Some(foreground) = raw.get("foreground").and_then(Value::as_str) {
                    style = style.fg(rgb(foreground, Color::Reset));
                }
                if let Some(background) = raw.get("background").and_then(Value::as_str) {
                    style = style.bg(rgb(background, Color::Reset));
                }
                if let Some(modifiers) = raw.get("modifiers").and_then(Value::as_array) {
                    for modifier in modifiers.iter().filter_map(Value::as_str) {
                        style = style.add_modifier(match modifier {
                            "bold" => Modifier::BOLD,
                            "dim" => Modifier::DIM,
                            "italic" => Modifier::ITALIC,
                            "underlined" => Modifier::UNDERLINED,
                            "reversed" => Modifier::REVERSED,
                            "crossed_out" => Modifier::CROSSED_OUT,
                            _ => Modifier::empty(),
                        });
                    }
                }
                theme.styles.insert(key.clone(), style);
            }
        }
        theme
    }

    pub fn base(&self) -> Style {
        self.styles
            .get("base")
            .copied()
            .unwrap_or_else(|| Style::default().fg(self.text).bg(self.panel))
    }

    pub fn selected(&self) -> Style {
        self.styles.get("selection").copied().unwrap_or_else(|| {
            Style::default()
                .fg(self.text)
                .bg(self.selection)
                .add_modifier(Modifier::BOLD)
        })
    }

    pub fn border_type(&self) -> BorderType {
        match self.border_set.as_str() {
            "plain" => BorderType::Plain,
            "double" => BorderType::Double,
            "thick" => BorderType::Thick,
            _ => BorderType::Rounded,
        }
    }
}

fn text(value: &Value, pointer: &str, fallback: &str) -> String {
    value
        .pointer(pointer)
        .and_then(Value::as_str)
        .unwrap_or(fallback)
        .to_owned()
}

fn color(value: &Value, pointer: &str, fallback: Color) -> Color {
    value
        .pointer(pointer)
        .and_then(Value::as_str)
        .map(|raw| rgb(raw, fallback))
        .unwrap_or(fallback)
}

fn rgb(raw: &str, fallback: Color) -> Color {
    let value = raw.trim().trim_start_matches('#');
    if value.len() != 6 {
        return fallback;
    }
    let Ok(red) = u8::from_str_radix(&value[0..2], 16) else {
        return fallback;
    };
    let Ok(green) = u8::from_str_radix(&value[2..4], 16) else {
        return fallback;
    };
    let Ok(blue) = u8::from_str_radix(&value[4..6], 16) else {
        return fallback;
    };
    Color::Rgb(red, green, blue)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn legacy_and_ratatui_theme_fields_merge() {
        let theme = Theme::from_value(&json!({
            "id":"custom","title":"Custom",
            "theme":{"background":"#010203","secondary":"#112233"},
            "colors":{"accent":"#aabbcc","text":"#ffffff","panel_bg":"#000000"},
            "ratatui":{"border_set":"double","syntax_theme":"base16-ocean.dark","styles":{
                "selection":{"foreground":"#ffffff","background":"#222222","modifiers":["bold"]}
            }}
        }));
        assert_eq!(theme.id, "custom");
        assert_eq!(theme.accent, Color::Rgb(0xaa, 0xbb, 0xcc));
        assert_eq!(theme.border_type(), BorderType::Double);
        assert!(theme.styles.contains_key("selection"));
    }
}
