import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class TranscriptStore:
    """Append-only JSONL transcript store for session resume and auditing."""

    def __init__(self, root_dir: str = "data/transcripts", enabled: bool = False):
        self.root_dir = Path(root_dir)
        self.enabled = enabled

    def append(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        ts: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        event = {
            "ts": ts if ts is not None else time.time(),
            "session_id": session_id,
            "role": role,
            "content": content,
            "metadata": metadata or {},
        }
        self.root_dir.mkdir(parents=True, exist_ok=True)
        with self._path_for(session_id).open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event

    def load(self, session_id: str) -> List[Dict[str, Any]]:
        path = self._path_for(session_id)
        if not path.exists():
            return []
        events = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                events.append(json.loads(line))
        return events

    def list_sessions(self) -> List[Dict[str, Any]]:
        if not self.root_dir.exists():
            return []
        sessions = []
        for path in sorted(self.root_dir.glob("*.jsonl")):
            session_id = path.stem
            events = self.load(session_id)
            if not events:
                continue
            summary_event = self._latest_summary_event(events)
            sessions.append({
                "session_id": session_id,
                "title": self._session_title(session_id, events, summary_event),
                "event_count": len(events),
                "updated_at": events[-1].get("ts"),
                "last_role": events[-1].get("role"),
                "last_content": self._last_display_content(events),
            })
        return sorted(sessions, key=lambda item: item.get("updated_at") or 0, reverse=True)

    def delete(self, session_id: str) -> bool:
        path = self._path_for(session_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def rename(self, old_session_id: str, new_session_id: str) -> bool:
        old_path = self._path_for(old_session_id)
        new_path = self._path_for(new_session_id)
        if not old_path.exists():
            return False
        self.root_dir.mkdir(parents=True, exist_ok=True)
        if new_path.exists():
            raise FileExistsError(f"目标 session 已存在：{new_session_id}")
        old_path.rename(new_path)
        events = self.load(new_session_id)
        if events:
            with new_path.open("w", encoding="utf-8") as file:
                for event in events:
                    event["session_id"] = new_session_id
                    file.write(json.dumps(event, ensure_ascii=False) + "\n")
        return True

    def replace(self, session_id: str, events: List[Dict[str, Any]]) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        with self._path_for(session_id).open("w", encoding="utf-8") as file:
            for event in events:
                event = dict(event)
                event["session_id"] = session_id
                file.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _path_for(self, session_id: str) -> Path:
        safe_session_id = re.sub(r"[^A-Za-z0-9_.-]", "_", session_id or "default")
        return self.root_dir / f"{safe_session_id}.jsonl"

    def _latest_summary_event(self, events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        for event in reversed(events):
            if event.get("role") == "summary":
                return event
        return None

    def _session_title(
        self,
        session_id: str,
        events: List[Dict[str, Any]],
        summary_event: Optional[Dict[str, Any]],
    ) -> str:
        if summary_event:
            metadata = summary_event.get("metadata") or {}
            title = str(metadata.get("title") or "").strip()
            if title:
                return title
            content = str(summary_event.get("content") or "").strip()
            if content:
                return self._compact_title(content)
        for event in events:
            if event.get("role") == "user":
                content = str(event.get("content") or "").strip()
                if content:
                    return self._compact_title(content)
        return session_id

    def _last_display_content(self, events: List[Dict[str, Any]]) -> str:
        for event in reversed(events):
            if event.get("role") in {"user", "assistant"}:
                return self._compact_preview(str(event.get("content") or ""))
        return ""

    def _compact_title(self, text: str, max_chars: int = 28) -> str:
        text = re.sub(r"\s+", " ", text or "").strip()
        text = re.sub(r"^(user|assistant|summary)\s*:\s*", "", text, flags=re.IGNORECASE)
        if not text:
            return ""
        sentence = re.split(r"[。！？!?。\n]", text, maxsplit=1)[0].strip() or text
        return sentence[: max_chars - 1] + "…" if len(sentence) > max_chars else sentence

    def _compact_preview(self, text: str, max_chars: int = 48) -> str:
        text = re.sub(r"\s+", " ", text or "").strip()
        return text[: max_chars - 1] + "…" if len(text) > max_chars else text
