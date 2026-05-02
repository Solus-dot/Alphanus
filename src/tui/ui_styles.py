ALPHANUS_TUI_CSS = """
    Screen {
        layout: vertical;
        background: $background;
        color: $foreground;
    }

    #topbar {
        height: 3;
        layout: horizontal;
        background: $panel;
        border-bottom: solid $app-border;
        padding: 0 2;
    }

    #topbar-left {
        width: 1fr;
        height: 3;
        content-align: left middle;
    }

    #topbar-center {
        width: auto;
        min-width: 0;
        height: 3;
        content-align: left middle;
    }

    #topbar-right {
        width: auto;
        height: 3;
        content-align: right middle;
        padding-left: 2;
    }

    #main-area {
        height: 1fr;
        layout: horizontal;
        background: $panel;
    }

    #chat-column {
        width: 1fr;
        height: 1fr;
        layout: vertical;
        background: $panel;
    }

    #chat-scroll {
        width: 1fr;
        height: 1fr;
        background: $panel;
        overflow-x: hidden;
        scrollbar-size: 1 1;
        scrollbar-color: $app-border $panel;
        scrollbar-background: $panel;
        scrollbar-background-hover: $surface;
        scrollbar-background-active: $surface;
        scrollbar-color-hover: $app-border;
        scrollbar-color-active: $accent;
        scrollbar-corner-color: $panel;
    }

    #chat-scroll.-active-panel {
        border: round $accent;
    }

    #chat-log {
        width: 1fr;
        height: auto;
        background: $panel;
        padding: 0 3 0 1;
        overflow-x: hidden;
        scrollbar-size: 0 0;
    }

    #partial {
        width: 1fr;
        height: auto;
        background: $panel;
        display: none;
        padding: 0 3 0 1;
        overflow-x: hidden;
    }

    #sidebar {
        width: 38;
        border-left: solid $app-border;
        background: $panel;
        display: none;
        padding: 0;
        layout: grid;
        grid-size: 1 2;
        grid-rows: 3fr 2fr;
    }

    #sidebar-footer-sep {
        position: absolute;
        width: 100%;
        height: 1;
        background: $panel;
        border-top: solid $app-border;
        border-left: solid $app-border;
        color: $app-border;
        padding: 0 0;
    }

    #sidebar-tree-section {
        width: 1fr;
        height: 1fr;
        layout: vertical;
        min-height: 5;
    }

    #sidebar-tree-header,
    #sidebar-inspector-header {
        width: 1fr;
        height: auto;
        color: $app-muted;
        text-style: bold;
        padding: 1 2 0 2;
    }

    #sidebar-tree-meta {
        width: 1fr;
        height: auto;
        color: $app-subtle;
        padding: 0 2 1 2;
    }

    #sidebar-tree-scroll {
        width: 1fr;
        height: 1fr;
        background: $panel;
        padding: 0 2 1 2;
        scrollbar-background: $panel;
        scrollbar-background-hover: $surface;
        scrollbar-background-active: $surface;
        scrollbar-color: $surface;
        scrollbar-color-hover: $app-border;
        scrollbar-color-active: $accent;
        scrollbar-corner-color: $panel;
    }

    #sidebar.-active-panel {
        border-left: solid $accent;
    }

    #sidebar-tree-content {
        width: 1fr;
        height: auto;
        background: $panel;
        text-wrap: nowrap;
    }

    #sidebar-inspector-section {
        width: 1fr;
        height: 1fr;
        min-height: 8;
        border-top: solid $app-border;
        background: $panel;
        layout: vertical;
    }

    #sidebar-inspector-scroll {
        width: 1fr;
        height: 1fr;
        background: $panel;
        padding: 0 2 1 2;
        scrollbar-background: $panel;
        scrollbar-background-hover: $surface;
        scrollbar-background-active: $surface;
        scrollbar-color: $surface;
        scrollbar-color-hover: $app-border;
        scrollbar-color-active: $accent;
        scrollbar-corner-color: $panel;
    }

#sidebar-inspector-content {
    width: 1fr;
    height: auto;
    background: $panel;
    text-wrap: nowrap;
}

    #footer {
        width: 1fr;
        height: 6;
        background: $panel;
        layout: vertical;
        padding: 0 0 0 1;
    }

    #command-popup {
        width: 64;
        max-height: 13;
        background: $panel;
        border: round $app-border;
        display: none;
        position: absolute;
        overlay: screen;
        padding: 0 0;
    }

    #command-popup-title {
        height: auto;
        color: $accent;
        padding: 0 0 0 0;
        text-style: bold;
    }

    #command-popup-hint {
        height: auto;
        color: $app-muted;
        padding: 0 0 1 0;
    }

    #command-options {
        width: 1fr;
        height: auto;
        max-height: 8;
        background: $panel;
        border: none;
        padding: 0 0 0 0;
        scrollbar-size: 1 1;
        scrollbar-background: $panel;
        scrollbar-background-hover: $surface;
        scrollbar-background-active: $surface;
        scrollbar-color: $surface;
        scrollbar-color-hover: $app-border;
        scrollbar-color-active: $accent;
        scrollbar-corner-color: $panel;
    }

    #command-options > .option-list--option-highlighted {
        color: $foreground;
        background: $surface;
        text-style: none;
    }

    #command-options:focus > .option-list--option-highlighted {
        color: $foreground;
        background: $app-selection-bg;
        text-style: bold;
    }

    #footer-sep {
        width: 1fr;
        height: 1;
        background: $panel;
        border-top: solid $app-border;
        color: $app-border;
        padding: 0 0;
    }

    #attachment-bar {
        width: 1fr;
        height: 1;
        background: $panel;
        color: $foreground;
        content-align: left middle;
        padding: 0 1;
    }

    #status-bar {
        width: 1fr;
        height: 1;
        layout: horizontal;
        padding: 0 1;
        background: $panel;
    }

    #status-left {
        width: 1fr;
        height: 1;
        content-align: left middle;
    }

    #status-right {
        width: auto;
        height: 1;
        content-align: right middle;
    }

    #input-row {
        width: 1fr;
        height: 3;
        layout: vertical;
        background: $panel;
        padding: 0 0 0 0;
        min-height: 3;
    }

    #composer-shell {
        width: 1fr;
        height: 3;
        layout: horizontal;
        background: $panel;
        border: round $app-border;
        padding: 0 1;
        margin: 0 0 0 0;
        align: left middle;
    }

    ChatInput {
        width: 1fr;
        height: 3;
        border: none;
        background: transparent;
        color: $foreground;
        padding: 0 0;
    }

    ChatInput:focus {
        border: none;
        background: transparent;
    }

    #input-row.-active-panel #composer-shell {
        border: round $accent;
    }

    #input-accessories {
        width: auto;
        height: 1;
        layout: horizontal;
        align: right middle;
        padding-left: 1;
    }

    #attach-file {
        width: auto;
        min-width: 8;
        height: 1;
        background: $app-selection-bg;
        color: $accent;
        border: none;
        text-style: bold;
        padding: 0 0;
    }
    """
