from __future__ import annotations

import queue
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(slots=True)
class StreamRuntimeState:
    event_queue: queue.SimpleQueue[Dict[str, object]] = field(default_factory=queue.SimpleQueue)
    drain_active: bool = False
    partial_dirty: bool = False
    deferred_live_preview: Optional[Tuple[List[str], Optional[str]]] = None
