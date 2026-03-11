#!/usr/bin/env python3
from __future__ import annotations

import imaplib
import os
import sys

from core.tool_script import emit, read_args, read_config


def main() -> int:
    args = read_args()
    config = read_config()
    if not config.get("capabilities", {}).get("email_enabled", False):
        raise PermissionError("Email capability is disabled")

    user = os.getenv("EMAIL_USER")
    pw = os.getenv("EMAIL_PASSWORD")
    if not user or not pw:
        raise ValueError("EMAIL_USER or EMAIL_PASSWORD missing")

    host = config.get("capabilities", {}).get("email_imap_server", "imap.gmail.com")
    count = int(args.get("count", 5))
    with imaplib.IMAP4_SSL(host) as mail:
        mail.login(user, pw)
        mail.select("INBOX")
        _, ids = mail.search(None, "ALL")
        msg_ids = ids[0].split()[-count:]
        out = [m.decode("utf-8", errors="replace") for m in msg_ids]

    emit({"message_ids": out})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
