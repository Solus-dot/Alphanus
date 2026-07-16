use serde_json::Value;

const LIMITS: (usize, usize) = (8_000, 140);

#[derive(Debug, Clone, Default)]
pub(crate) struct ToolPreview {
    pub(crate) filepath: String,
    pub(crate) content: String,
    pub(crate) language: String,
    pub(crate) truncated: bool,
}

pub(crate) fn canonical_name(name: &str) -> &str {
    name.rsplit([':', '.']).next().unwrap_or(name)
}

pub(crate) fn language_for_filepath(filepath: &str) -> &str {
    let extension = filepath
        .rsplit_once('.')
        .map_or(filepath, |parts| parts.1)
        .to_ascii_lowercase();
    match extension.as_str() {
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

pub(crate) fn bounded(content: &str) -> (String, bool) {
    let lines = content.lines().collect::<Vec<_>>();
    let mut output = lines
        .iter()
        .take(LIMITS.1)
        .copied()
        .collect::<Vec<_>>()
        .join("\n");
    let mut truncated = lines.len() > LIMITS.1;
    if output.len() > LIMITS.0 {
        let boundary = output
            .char_indices()
            .map(|(index, _)| index)
            .take_while(|index| *index <= LIMITS.0)
            .last()
            .unwrap_or(0);
        output.truncate(boundary);
        truncated = true;
    }
    (output, truncated)
}

pub(crate) fn from_request(name: &str, data: &Value) -> ToolPreview {
    if !matches!(canonical_name(name), "create_file" | "edit_file") {
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
    let (content, truncated) = bounded(content);
    ToolPreview {
        filepath,
        content,
        language: String::new(),
        truncated,
    }
}

pub(crate) fn from_result(name: &str, data: &Value) -> ToolPreview {
    let canonical = canonical_name(name);
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
    let (content, locally_truncated) = bounded(content);
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

pub(crate) fn partial_json_string(raw: &str, key: &str) -> Option<String> {
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
