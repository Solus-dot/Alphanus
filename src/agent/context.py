from __future__ import annotations

from typing import cast

from core.message_types import ChatMessage, JSONValue, MessageContentPart, ToolCallDelta, ToolFunctionCall


class ContextWindowManager:
    _MEDIA_PART_CHAR_COST = 256
    _STRUCTURED_PART_OVERHEAD = 32

    def __init__(self, context_limit: int = 8192, keep_last_n: int = 10, safety_margin: int = 500):
        self.context_limit = int(context_limit)
        self.keep_last_n = int(keep_last_n)
        self.safety_margin = int(safety_margin)

    @staticmethod
    def _estimate_message_chars(message: ChatMessage) -> int:
        chars = len(message.get("role", "")) + 4
        content = message.get("content", "")
        if isinstance(content, str):
            chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    chars += len(str(part))
                    continue
                part_type = str(part.get("type", "")).strip().lower()
                if part_type == "text":
                    chars += len(str(part.get("text", "")))
                    continue
                if part_type in {"image", "image_url", "video"}:
                    chars += ContextWindowManager._MEDIA_PART_CHAR_COST
                    continue
                chars += ContextWindowManager._STRUCTURED_PART_OVERHEAD
        else:
            chars += len(str(content))
        raw_tool_calls = message.get("tool_calls")
        if raw_tool_calls:
            chars += len(str(raw_tool_calls))
        return chars

    @staticmethod
    def _chars_to_tokens(chars: int) -> int:
        return max(1, chars // 4)

    @classmethod
    def estimate_tokens(cls, messages: list[ChatMessage]) -> int:
        return cls._chars_to_tokens(sum(cls._estimate_message_chars(message) for message in messages))

    @staticmethod
    def _last_role_index(messages: list[ChatMessage], role: str) -> int | None:
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role") == role:
                return idx
        return None

    @staticmethod
    def _content_to_user_text(content: JSONValue | list[MessageContentPart]) -> str:
        if isinstance(content, str):
            text = content.strip()
            return text if text else "[user message omitted]"
        if isinstance(content, list):
            chunks: list[str] = []
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
    def _compact_user_content(
        cls,
        content: JSONValue | list[MessageContentPart],
        keep_chars: int,
    ) -> JSONValue | list[MessageContentPart]:
        if isinstance(content, list):
            compacted_parts: list[MessageContentPart] = []
            remaining_chars = max(0, keep_chars)
            omitted_text = False
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = str(part.get("type", "")).strip().lower()
                if part_type == "text":
                    text = str(part.get("text", "")).strip()
                    if not text:
                        continue
                    if remaining_chars > 0:
                        text = cls._truncate_text(text, remaining_chars)
                        remaining_chars = max(0, remaining_chars - len(text))
                        compacted_parts.append({"type": "text", "text": text})
                        continue
                    if not omitted_text:
                        compacted_parts.append({"type": "text", "text": "[user message omitted]"})
                        omitted_text = True
                    continue
                compacted_parts.append(cast(MessageContentPart, dict(part)))
            if compacted_parts:
                return compacted_parts

        text = cls._content_to_user_text(content)
        if keep_chars > 0:
            text = cls._truncate_text(text, keep_chars)
        return text or "[user message omitted]"

    @classmethod
    def _compact_user_message(cls, message: ChatMessage, keep_chars: int) -> ChatMessage:
        return {"role": "user", "content": cls._compact_user_content(message.get("content"), keep_chars)}

    @staticmethod
    def _tool_call_ids(message: ChatMessage) -> set[str]:
        raw_calls = message.get("tool_calls") or []
        if not isinstance(raw_calls, list):
            return set()
        out: set[str] = set()
        for call in raw_calls:
            if isinstance(call, dict):
                call_id = str(call.get("id") or "").strip()
                if call_id:
                    out.add(call_id)
        return out

    @classmethod
    def _find_matching_assistant_idx(
        cls,
        messages: list[ChatMessage],
        upto_idx: int,
        tool_call_id: str,
    ) -> int | None:
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
    def _expand_tool_dependencies(cls, messages: list[ChatMessage], kept: set[int]) -> set[int]:
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
    def _compress_messages_to_budget(cls, messages: list[ChatMessage], max_prompt_tokens: int) -> list[ChatMessage]:
        if max_prompt_tokens <= 0:
            return messages[:1]

        out: list[ChatMessage] = []
        for message in messages:
            copied = cast(ChatMessage, dict(message))
            raw_tool_calls = copied.get("tool_calls")
            if isinstance(raw_tool_calls, list):
                copied_calls: list[ToolCallDelta] = []
                for call in raw_tool_calls:
                    if not isinstance(call, dict):
                        continue
                    call_copy = cast(ToolCallDelta, dict(call))
                    fn = call_copy.get("function")
                    if isinstance(fn, dict):
                        call_copy["function"] = cast(ToolFunctionCall, dict(fn))
                    copied_calls.append(call_copy)
                copied["tool_calls"] = copied_calls
            out.append(copied)
        estimates = [cls._estimate_message_chars(message) for message in out]
        total_estimated = sum(estimates)

        def shrink_once() -> bool:
            nonlocal total_estimated
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
                previous = estimates[best_msg_idx]
                estimates[best_msg_idx] = cls._estimate_message_chars(out[best_msg_idx])
                total_estimated += estimates[best_msg_idx] - previous
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
            previous = estimates[best_msg_idx]
            estimates[best_msg_idx] = cls._estimate_message_chars(out[best_msg_idx])
            total_estimated += estimates[best_msg_idx] - previous
            return True

        while cls._chars_to_tokens(total_estimated) > max_prompt_tokens:
            if not shrink_once():
                break

        def required_indexes() -> set[int]:
            required: set[int] = set()
            required_tool_ids: set[str] = set()
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

        while cls._chars_to_tokens(total_estimated) > max_prompt_tokens and len(out) > 1:
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
                total_estimated -= estimates.pop(drop_idx)
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
                break

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

            total_estimated -= estimates.pop(tool_idx)
            out.pop(tool_idx)
            if assistant_idx is not None:
                if assistant_idx > tool_idx:
                    assistant_idx -= 1
                total_estimated -= estimates.pop(assistant_idx)
                out.pop(assistant_idx)

        if cls._chars_to_tokens(total_estimated) > max_prompt_tokens and out:
            sys_content = out[0].get("content")
            if isinstance(sys_content, str) and sys_content:
                keep_chars = max(32, max_prompt_tokens * 2)
                out[0]["content"] = cls._truncate_text(sys_content, keep_chars)
                estimates[0] = cls._estimate_message_chars(out[0])
                total_estimated = sum(estimates)

        if cls._chars_to_tokens(total_estimated) > max_prompt_tokens:
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

            return [out[0], cls._compact_user_message(out[last_user], keep_chars=16)]
        return out

    def prune(self, messages: list[ChatMessage], max_tokens: int) -> list[ChatMessage]:
        if len(messages) <= 2:
            return messages

        budget = self.context_limit - self.safety_margin
        message_estimates = [self._estimate_message_chars(message) for message in messages]
        estimated = self._chars_to_tokens(sum(message_estimates))
        if estimated + max_tokens <= budget:
            return messages

        keep_tail = self.keep_last_n
        if len(messages) <= keep_tail + 1:
            return messages

        head = messages[:1]
        body = messages[1:]
        head_estimate = message_estimates[0]
        body_estimates = message_estimates[1:]

        start_idx = max(0, len(body) - keep_tail)
        kept_indices: set[int] = set(range(start_idx, len(body)))
        kept_indices = self._expand_tool_dependencies(body, kept_indices)
        last_user_idx = self._last_role_index(body, "user")
        if last_user_idx is not None:
            kept_indices.add(last_user_idx)

        selected = [body[idx] for idx in sorted(kept_indices)]
        selected_total = head_estimate + sum(body_estimates[idx] for idx in kept_indices)

        # Ensure exactly one system message at head to satisfy backend template expectations.
        pruned = head + selected
        if self._chars_to_tokens(selected_total) + max_tokens <= budget:
            return pruned

        # If still over budget, trim oldest non-dependent messages first.
        required_assistant_idxs = set()
        required_tool_ids = {str(body[idx].get("tool_call_id") or "").strip() for idx in kept_indices if body[idx].get("role") == "tool"}
        required_tool_ids.discard("")
        for idx in kept_indices:
            if body[idx].get("role") != "assistant":
                continue
            call_ids = self._tool_call_ids(body[idx])
            if call_ids & required_tool_ids:
                required_assistant_idxs.add(idx)

        required_idxs = {idx for idx in kept_indices if body[idx].get("role") == "tool"} | required_assistant_idxs
        if last_user_idx is not None:
            required_idxs.add(last_user_idx)

        mutable = [idx for idx in sorted(kept_indices)]
        drop_cursor = 0
        current_total = selected_total
        while drop_cursor < len(mutable):
            if self._chars_to_tokens(current_total) + max_tokens <= budget:
                break
            idx = mutable[drop_cursor]
            if idx in required_idxs:
                drop_cursor += 1
                continue
            current_total -= body_estimates[idx]
            del mutable[drop_cursor]

        candidate = head if not mutable else head + [body[idx] for idx in mutable]
        max_prompt_tokens = max(1, budget - max_tokens)
        candidate_total = head_estimate + sum(body_estimates[idx] for idx in mutable) if mutable else head_estimate
        if self._chars_to_tokens(candidate_total) <= max_prompt_tokens and self.estimate_tokens(candidate) <= max_prompt_tokens:
            return candidate
        return self._compress_messages_to_budget(candidate, max_prompt_tokens)
