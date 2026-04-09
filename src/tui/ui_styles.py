ALPHANUS_TUI_CSS = """
    Screen {
        layout: vertical;
        background: #09090b;
        color: #e4e4e7;
    }

    #topbar {
        height: 3;
        layout: horizontal;
        background: #000000;
        border-bottom: solid #52525b;
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
        background: #09090b;
    }

    #chat-column {
        width: 1fr;
        height: 1fr;
        layout: vertical;
        background: #09090b;
    }

    #chat-scroll {
        width: 1fr;
        height: 1fr;
        background: #09090b;
        overflow-x: hidden;
        scrollbar-size: 1 1;
        scrollbar-color: #52525b #000000;
        scrollbar-background: #000000;
        scrollbar-background-hover: #18181b;
        scrollbar-background-active: #18181b;
        scrollbar-color-hover: #52525b;
        scrollbar-color-active: #6366f1;
        scrollbar-corner-color: #000000;
    }

    #chat-scroll.-active-panel {
        border: round #6366f1;
    }

    #chat-log {
        width: 1fr;
        height: auto;
        background: #09090b;
        padding: 0 3 0 1;
        overflow-x: hidden;
        scrollbar-size: 0 0;
    }

    #partial {
        width: 1fr;
        height: auto;
        background: #09090b;
        display: none;
        padding: 0 3 0 1;
        overflow-x: hidden;
    }

    #sidebar {
        width: 38;
        border-left: solid #52525b;
        background: #000000;
        display: none;
        padding: 0;
        layout: grid;
        grid-size: 1 2;
        grid-rows: 3fr 2fr;
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
        color: #a1a1aa;
        text-style: bold;
        padding: 1 2 0 2;
    }

    #sidebar-tree-meta {
        width: 1fr;
        height: auto;
        color: #71717a;
        padding: 0 2 1 2;
    }

    #sidebar-tree-scroll {
        width: 1fr;
        height: 1fr;
        background: #000000;
        padding: 0 2 1 2;
        scrollbar-background: #000000;
        scrollbar-background-hover: #18181b;
        scrollbar-background-active: #18181b;
        scrollbar-color: #3f3f46;
        scrollbar-color-hover: #52525b;
        scrollbar-color-active: #6366f1;
        scrollbar-corner-color: #000000;
    }

    #sidebar.-active-panel {
        border-left: solid #6366f1;
    }

    #sidebar-tree-content {
        width: 1fr;
        height: auto;
        background: #000000;
    }

    #sidebar-inspector-section {
        width: 1fr;
        height: 1fr;
        min-height: 8;
        border-top: solid #52525b;
        background: #000000;
        layout: vertical;
    }

    #sidebar-inspector-scroll {
        width: 1fr;
        height: 1fr;
        background: #000000;
        padding: 0 2 1 2;
        scrollbar-background: #000000;
        scrollbar-background-hover: #18181b;
        scrollbar-background-active: #18181b;
        scrollbar-color: #3f3f46;
        scrollbar-color-hover: #52525b;
        scrollbar-color-active: #6366f1;
        scrollbar-corner-color: #000000;
    }

    #sidebar-inspector-content {
        width: 1fr;
        height: auto;
        background: #000000;
    }

    #footer {
        width: 1fr;
        height: 7;
        background: #09090b;
        layout: vertical;
        padding: 0 3 0 3;
    }

    #command-popup {
        width: 64;
        max-height: 13;
        background: #000000;
        border: round #52525b;
        display: none;
        position: absolute;
        overlay: screen;
        padding: 0 1;
    }

    #command-popup-title {
        height: auto;
        color: #6366f1;
        padding: 1 1 0 1;
        text-style: bold;
    }

    #command-popup-hint {
        height: auto;
        color: #a1a1aa;
        padding: 0 1 1 1;
    }

    #command-options {
        width: 1fr;
        height: auto;
        max-height: 8;
        background: #000000;
        border: none;
        padding: 0 1 1 1;
        scrollbar-background: #000000;
        scrollbar-background-hover: #18181b;
        scrollbar-background-active: #18181b;
        scrollbar-color: #3f3f46;
        scrollbar-color-hover: #52525b;
        scrollbar-color-active: #6366f1;
        scrollbar-corner-color: #000000;
    }

    #command-options > .option-list--option-highlighted {
        color: #e4e4e7;
        background: #18181b;
        text-style: none;
    }

    #command-options:focus > .option-list--option-highlighted {
        color: #ffffff;
        background: #1a1730;
        text-style: bold;
    }

    #footer-sep {
        height: 1;
        background: #000000;
        color: #5a5a66;
        content-align: left middle;
        padding: 0 0;
    }

    #attachment-bar {
        width: 1fr;
        height: 1;
        background: #000000;
        color: #e4e4e7;
        content-align: left middle;
        padding: 0 1;
    }

    #status-bar {
        height: 1;
        layout: horizontal;
        padding: 0 0;
        background: #09090b;
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
        height: 4;
        layout: vertical;
        background: #09090b;
        padding: 0 0 0 0;
        min-height: 4;
    }

    #composer-shell {
        width: 1fr;
        height: 3;
        layout: horizontal;
        background: #000000;
        border: round #63636b;
        padding: 0 1;
        margin: 0 0 1 0;
        align: left middle;
    }

    ChatInput {
        width: 1fr;
        height: 3;
        border: none;
        background: transparent;
        color: #e4e4e7;
    }

    ChatInput:focus {
        border: none;
        background: transparent;
    }

    #input-row.-active-panel #composer-shell {
        border: round #6366f1;
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
        background: #1a1730;
        color: #6366f1;
        border: none;
        text-style: bold;
        padding: 0 1;
    }
    """
