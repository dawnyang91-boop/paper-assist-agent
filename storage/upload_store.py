import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, List, Optional

from config import AppConfig, get_config


class UploadStore:
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or get_config()
        self.root = Path(self.config.rag_upload_dir)
        if not self.root.is_absolute():
            self.root = Path(__file__).resolve().parent / self.root
        self.root.mkdir(parents=True, exist_ok=True)

    def allowed_extensions(self) -> set[str]:
        return {
            item.strip().lower()
            for item in self.config.rag_upload_allowed_extensions.split(",")
            if item.strip()
        }

    def save_files(self, files: Iterable[Any], user_id: str = "local") -> List[dict]:
        saved = []
        for file in files:
            original_name = Path(file.filename or "upload").name
            ext = Path(original_name).suffix.lower()
            if ext not in self.allowed_extensions():
                raise ValueError(f"不支持的文件类型：{ext or original_name}")
            upload_id = str(uuid.uuid4())
            target_dir = self.root / self._safe_part(user_id)
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"{upload_id}{ext}"
            size = self._copy_upload(file, target)
            max_size = self.config.rag_upload_max_file_size_mb * 1024 * 1024
            if size > max_size:
                target.unlink(missing_ok=True)
                raise ValueError(f"文件超过大小限制：{original_name}")
            saved.append({
                "file_id": upload_id,
                "filename": original_name,
                "path": str(target),
                "size": size,
                "created_at": time.time(),
            })
        return saved

    def replace_files(self, files: Iterable[Any], user_id: str = "local") -> List[dict]:
        """Replace the pending upload queue with the newly selected files."""
        self.clear_files(user_id=user_id)
        return self.save_files(files, user_id=user_id)

    def list_files(self, user_id: str = "local") -> List[dict]:
        target_dir = self.root / self._safe_part(user_id)
        if not target_dir.exists():
            return []
        results = []
        for path in sorted(target_dir.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
            if not path.is_file():
                continue
            results.append({
                "file_id": path.stem,
                "filename": path.name,
                "path": str(path),
                "size": path.stat().st_size,
                "created_at": path.stat().st_mtime,
            })
        return results

    def delete_file(self, file_id: str, user_id: str = "local") -> bool:
        target_dir = self.root / self._safe_part(user_id)
        for path in target_dir.glob(f"{file_id}.*"):
            if path.is_file():
                path.unlink()
                return True
        return False

    def clear_files(self, user_id: str = "local") -> int:
        target_dir = self.root / self._safe_part(user_id)
        if not target_dir.exists():
            return 0
        deleted = 0
        for path in target_dir.iterdir():
            if path.is_file():
                path.unlink()
                deleted += 1
        return deleted

    def upload_dir(self, user_id: str = "local") -> str:
        path = self.root / self._safe_part(user_id)
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def _copy_upload(self, file: Any, target: Path) -> int:
        with target.open("wb") as output:
            shutil.copyfileobj(file.file, output)
        return os.path.getsize(target)

    def _safe_part(self, value: str) -> str:
        return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value or "local")
