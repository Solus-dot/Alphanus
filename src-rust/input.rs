use crossterm::event::{KeyCode, KeyEvent};

pub(crate) fn edit(value: &mut String, cursor: &mut usize, key: KeyEvent, multiline: bool) -> bool {
    let previous = || {
        value[..*cursor]
            .char_indices()
            .next_back()
            .map_or(0, |(index, _)| index)
    };
    let next = || {
        value[*cursor..]
            .char_indices()
            .nth(1)
            .map_or(value.len(), |(index, _)| *cursor + index)
    };
    match key.code {
        KeyCode::Char(character) => {
            value.insert(*cursor, character);
            *cursor += character.len_utf8();
        }
        KeyCode::Enter if multiline => {
            value.insert(*cursor, '\n');
            *cursor += 1;
        }
        KeyCode::Backspace if *cursor > 0 => {
            let start = previous();
            value.drain(start..*cursor);
            *cursor = start;
        }
        KeyCode::Delete if *cursor < value.len() => {
            value.drain(*cursor..next());
        }
        KeyCode::Left => *cursor = previous(),
        KeyCode::Right => *cursor = next(),
        KeyCode::Home => *cursor = 0,
        KeyCode::End => *cursor = value.len(),
        _ => return false,
    }
    true
}
