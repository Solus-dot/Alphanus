from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class WhatsAppMessage:
    provider_message_id: str
    chat_id: str
    text: str
    timestamp_utc: datetime


class WhatsAppBridge:
    """Optional integration entrypoint.

    This module intentionally does not implement provider transport.
    It defines the contract Alphanus uses when a provider adapter is added.
    """

    @staticmethod
    def branch_label(chat_id: str, ts: datetime | None = None) -> str:
        ts = ts or datetime.now(timezone.utc)
        return f"wa:{chat_id}:{ts.strftime('%Y%m%d')}"
