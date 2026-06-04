import time
from typing import Any, Dict, Iterable, List


class SessionSummaryBuilder:
    """Build a compact memory item from recent transcript events."""

    def __init__(self, max_chars: int = 1200):
        self.max_chars = max_chars

    def build_memory(self, events: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        events_list = list(events)
        lines: List[str] = []
        for event in events_list:
            role = event.get("role", "transcript")
            content = str(event.get("content", "")).strip()
            if not content:
                continue
            lines.append(f"{role}: {content}")
        summary = "\n".join(lines)
        if len(summary) > self.max_chars:
            summary = summary[: self.max_chars - 3] + "..."
        return {
            "role": "transcript:summary",
            "type": "transcript_summary",
            "content": summary,
            "metadata": {
                "source": "transcript",
                "event_count": len(events_list),
                "title": self.build_title(events_list),
            },
        }

    def build_event(self, session_id: str, events: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        memory = self.build_memory(events)
        return {
            "ts": time.time(),
            "session_id": session_id,
            "role": "summary",
            "content": memory["content"],
            "metadata": memory["metadata"],
        }

    def build_title(self, events: Iterable[Dict[str, Any]], max_chars: int = 28) -> str:
        for event in events:
            if event.get("role") != "user":
                continue
            content = str(event.get("content", "")).strip()
            if not content:
                continue
            content = " ".join(content.split())
            sentence = content
            for separator in ["。", "！", "？", "!", "?", "\n"]:
                if separator in sentence:
                    sentence = sentence.split(separator, 1)[0]
                    break
            sentence = sentence.strip() or content
            return sentence[: max_chars - 1] + "…" if len(sentence) > max_chars else sentence
        return "未命名会话"
