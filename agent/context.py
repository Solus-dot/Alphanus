from typing import Any, Dict, List, Optional, Set


class ContextWindowManager:
    def __init__(self, context_limit: int = 8192, keep_last_n: int = 10, safety_margin: int = 500):
        self.context_limit = int(context_limit)
        self.keep_last_n = int(keep_last_n)
        self.safety_margin = int(safety_margin)

    @staticmethod
    def estimate_tokens(messages: List[Dict]) -> int:
        chars = 0
        for msg in messages:
            chars += len(msg.get("role", "")) + 4
            content = msg.get("content", "")
            if isinstance(content, str):
                chars += len(content)
            else:
                chars += len(str(content))
            if msg.get("tool_calls"):
                chars += len(str(msg["tool_calls"]))
        return max(1, chars // 4)

    @staticmethod
    def _last_role_index(messages: List[Dict], role: str) -> Optional[int]:
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role") == role:
                return idx
        return None

    @staticmethod
    def _content_to_user_text(content: Any) -> str:
        if isinstance(content, str):
            text = content.strip()
            return text if text else "[user message omitted]"
        if isinstance(content, list):
            chunks: List[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") != "text":
                    continue
                text = str(part.get("text", "")).strip()
                if text:
                    chunks.append(text)
            if chunks:
                return "\n".join(chunks)
        return "[user message omitted]"

    @classmethod
    def _compact_user_message(cls, message: Dict, keep_chars: int) -> Dict:
        text = cls._content_to_user_text(message.get("content"))
        if keep_chars > 0:
            text = cls._truncate_text(text, keep_chars)
        return {"role": "user", "content": text or "[user message omitted]"}

    @staticmethod
    def _tool_call_ids(message: Dict) -> Set[str]:
        raw_calls = message.get("tool_calls") or []
        if not isinstance(raw_calls, list):
            return set()
        out: Set[str] = set()
        for call in raw_calls:
            if isinstance(call, dict):
                call_id = str(call.get("id") or "").strip()
                if call_id:
                    out.add(call_id)
        return out

    @classmethod
    def _find_matching_assistant_idx(
        cls,
        messages: List[Dict],
        upto_idx: int,
        tool_call_id: str,
    ) -> Optional[int]:
        if not tool_call_id:
            return None
        for idx in range(upto_idx - 1, -1, -1):
            msg = messages[idx]
            if msg.get("role") != "assistant":
                continue
            if tool_call_id in cls._tool_call_ids(msg):
                return idx
        return None

    @classmethod
    def _expand_tool_dependencies(cls, messages: List[Dict], kept: Set[int]) -> Set[int]:
        changed = True
        while changed:
            changed = False
            snapshot = sorted(kept)
            for idx in snapshot:
                msg = messages[idx]
                if msg.get("role") != "tool":
                    continue
                tool_call_id = str(msg.get("tool_call_id") or "").strip()
                if not tool_call_id:
                    continue
                match_idx = cls._find_matching_assistant_idx(messages, idx, tool_call_id)
                if match_idx is not None and match_idx not in kept:
                    kept.add(match_idx)
                    changed = True
        return kept

    @staticmethod
    def _truncate_text(value: str, keep_chars: int) -> str:
        if keep_chars <= 0 or len(value) <= keep_chars:
            return value
        remaining = len(value) - keep_chars
        suffix = f"...[{remaining}c]"
        if keep_chars <= len(suffix):
            return value[:keep_chars]
        head = keep_chars - len(suffix)
        return value[:head] + suffix

    @classmethod
    def _compress_messages_to_budget(cls, messages: List[Dict], max_prompt_tokens: int) -> List[Dict]:
        if max_prompt_tokens <= 0:
            return messages[:1]

        out: List[Dict] = []
        for message in messages:
            copied = dict(message)
            raw_tool_calls = copied.get("tool_calls")
            if isinstance(raw_tool_calls, list):
                copied_calls = []
                for call in raw_tool_calls:
                    if not isinstance(call, dict):
                        copied_calls.append(call)
                        continue
                    call_copy = dict(call)
                    fn = call_copy.get("function")
                    if isinstance(fn, dict):
                        call_copy["function"] = dict(fn)
                    copied_calls.append(call_copy)
                copied["tool_calls"] = copied_calls
            out.append(copied)

        def shrink_once() -> bool:
            best_kind = None
            best_msg_idx = -1
            best_call_idx = -1
            best_len = 0

            for msg_idx, msg in enumerate(out):
                content = msg.get("content")
                if isinstance(content, str) and len(content) > 16 and len(content) > best_len:
                    best_kind = "content"
                    best_msg_idx = msg_idx
                    best_len = len(content)

                raw_calls = msg.get("tool_calls")
                if not isinstance(raw_calls, list):
                    continue
                for call_idx, call in enumerate(raw_calls):
                    if not isinstance(call, dict):
                        continue
                    fn = call.get("function")
                    if not isinstance(fn, dict):
                        continue
                    arguments = fn.get("arguments")
                    if isinstance(arguments, str) and len(arguments) > 16 and len(arguments) > best_len:
                        best_kind = "tool_args"
                        best_msg_idx = msg_idx
                        best_call_idx = call_idx
                        best_len = len(arguments)

            if best_kind is None:
                return False

            next_len = max(16, best_len // 2)
            if best_kind == "content":
                current = str(out[best_msg_idx].get("content", ""))
                out[best_msg_idx]["content"] = cls._truncate_text(current, next_len)
                return True

            raw_calls = out[best_msg_idx].get("tool_calls")
            if not isinstance(raw_calls, list) or best_call_idx < 0 or best_call_idx >= len(raw_calls):
                return False
            call = raw_calls[best_call_idx]
            if not isinstance(call, dict):
                return False
            fn = call.get("function")
            if not isinstance(fn, dict):
                return False
            current_args = str(fn.get("arguments", ""))
            fn["arguments"] = cls._truncate_text(current_args, next_len)
            return True

        while cls.estimate_tokens(out) > max_prompt_tokens:
            if not shrink_once():
                break

        def required_indexes() -> Set[int]:
            required: Set[int] = set()
            required_tool_ids: Set[str] = set()
            for msg_idx, msg in enumerate(out):
                if msg.get("role") != "tool":
                    continue
                required.add(msg_idx)
                tool_call_id = str(msg.get("tool_call_id") or "").strip()
                if tool_call_id:
                    required_tool_ids.add(tool_call_id)
            for msg_idx, msg in enumerate(out):
                if msg.get("role") != "assistant":
                    continue
                if cls._tool_call_ids(msg) & required_tool_ids:
                    required.add(msg_idx)
            return required

        while cls.estimate_tokens(out) > max_prompt_tokens and len(out) > 1:
            required = required_indexes()
            last_user = cls._last_role_index(out, "user")
            if last_user is not None:
                required.add(last_user)

            drop_idx = None
            for msg_idx in range(1, len(out)):
                if msg_idx not in required:
                    drop_idx = msg_idx
                    break
            if drop_idx is not None:
                out.pop(drop_idx)
                continue

            # If every remaining non-system message is required, drop the oldest
            # complete tool bundle (tool + matching assistant tool_calls) first.
            tool_idx = None
            for msg_idx in range(1, len(out)):
                if out[msg_idx].get("role") == "tool":
                    tool_idx = msg_idx
                    break
            if tool_idx is None:
                out.pop(1)
                continue

            tool_call_id = str(out[tool_idx].get("tool_call_id") or "").strip()
            assistant_idx = None
            if tool_call_id:
                for msg_idx in range(tool_idx - 1, 0, -1):
                    msg = out[msg_idx]
                    if msg.get("role") != "assistant":
                        continue
                    if tool_call_id in cls._tool_call_ids(msg):
                        assistant_idx = msg_idx
                        break

            out.pop(tool_idx)
            if assistant_idx is not None:
                out.pop(assistant_idx)

        if cls.estimate_tokens(out) > max_prompt_tokens and out:
            sys_content = out[0].get("content")
            if isinstance(sys_content, str) and sys_content:
                keep_chars = max(32, max_prompt_tokens * 2)
                out[0]["content"] = cls._truncate_text(sys_content, keep_chars)

        if cls.estimate_tokens(out) > max_prompt_tokens:
            last_user = cls._last_role_index(out, "user")
            if last_user is None:
                return out[:1]

            compact_user = cls._compact_user_message(
                out[last_user],
                keep_chars=max(16, min(256, max_prompt_tokens * 2)),
            )
            minimal = [out[0], compact_user]
            if cls.estimate_tokens(minimal) <= max_prompt_tokens:
                return minimal

            compact_user["content"] = cls._truncate_text(str(compact_user.get("content", "")), 16)
            minimal = [out[0], compact_user]
            return minimal
        return out

    def prune(self, messages: List[Dict], max_tokens: int) -> List[Dict]:
        if len(messages) <= 2:
            return messages

        budget = self.context_limit - self.safety_margin
        estimated = self.estimate_tokens(messages)
        if estimated + max_tokens <= budget:
            return messages

        keep_tail = self.keep_last_n
        if len(messages) <= keep_tail + 1:
            return messages

        head = messages[:1]
        body = messages[1:]

        start_idx = max(0, len(body) - keep_tail)
        kept_indices: Set[int] = set(range(start_idx, len(body)))
        kept_indices = self._expand_tool_dependencies(body, kept_indices)
        last_user_idx = self._last_role_index(body, "user")
        if last_user_idx is not None:
            kept_indices.add(last_user_idx)

        selected = [body[idx] for idx in sorted(kept_indices)]

        # Keep a single system message at index 0 to avoid server-side template
        # failures on backends that require exactly one leading system role.
        pruned = head + selected
        if self.estimate_tokens(pruned) + max_tokens <= budget:
            return pruned

        # If still over budget, trim oldest non-dependent messages first.
        required_assistant_idxs = set()
        required_tool_ids = {
            str(body[idx].get("tool_call_id") or "").strip()
            for idx in kept_indices
            if body[idx].get("role") == "tool"
        }
        required_tool_ids.discard("")
        for idx in kept_indices:
            if body[idx].get("role") != "assistant":
                continue
            call_ids = self._tool_call_ids(body[idx])
            if call_ids & required_tool_ids:
                required_assistant_idxs.add(idx)

        required_idxs = {
            idx for idx in kept_indices if body[idx].get("role") == "tool"
        } | required_assistant_idxs
        if last_user_idx is not None:
            required_idxs.add(last_user_idx)

        mutable = [idx for idx in sorted(kept_indices)]
        drop_cursor = 0
        while drop_cursor < len(mutable):
            if self.estimate_tokens(head + [body[idx] for idx in mutable]) + max_tokens <= budget:
                break
            idx = mutable[drop_cursor]
            if idx in required_idxs:
                drop_cursor += 1
                continue
            del mutable[drop_cursor]

        candidate = head if not mutable else head + [body[idx] for idx in mutable]
        max_prompt_tokens = max(1, budget - max_tokens)
        if self.estimate_tokens(candidate) <= max_prompt_tokens:
            return candidate
        return self._compress_messages_to_budget(candidate, max_prompt_tokens)
