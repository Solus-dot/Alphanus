from __future__ import annotations

from typing import Final

ENDPOINT_MODE_AUTO: Final[str] = "auto"
ENDPOINT_MODE_RESPONSES: Final[str] = "responses"
ENDPOINT_MODE_CHAT: Final[str] = "chat"
ENDPOINT_MODES: Final[frozenset[str]] = frozenset({ENDPOINT_MODE_AUTO, ENDPOINT_MODE_RESPONSES, ENDPOINT_MODE_CHAT})
CONCRETE_ENDPOINT_MODES: Final[frozenset[str]] = frozenset({ENDPOINT_MODE_RESPONSES, ENDPOINT_MODE_CHAT})
