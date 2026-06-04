from pathlib import Path
from typing import Any, Iterable, List, Optional

from config import AppConfig, get_config


class EmbeddingService:
    """Unified text embedding client for RAG queries and future memory writes."""

    def __init__(self, config: Optional[AppConfig] = None, client: Any = None):
        self.config = config or get_config()
        self.client = client
        self.local_model = None

    def _uses_local_model(self) -> bool:
        model_type = (getattr(self.config, "embed_model_type", "") or "").lower().replace("_", "-")
        return (
            model_type in {"local", "sentence-transformers", "sentence-transformer"}
            or self.config.embed_model_name.startswith("sentence-transformers/")
        )

    def _get_client(self):
        if self.client is not None:
            return self.client
        if not self.config.embedding_api_key:
            raise ValueError("未配置 EMBED_API_KEY 或 OPENAI_API_KEY，无法生成查询向量。")

        from openai import OpenAI

        self.client = OpenAI(
            api_key=self.config.embedding_api_key,
            base_url=self.config.embedding_base_url,
        )
        return self.client

    def _get_local_model(self):
        if self.local_model is not None:
            return self.local_model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("未安装 sentence-transformers，无法使用本地 embedding。请运行 `pip install -r requirements.txt`。") from exc

        model_name = self._resolve_local_model_path(self.config.embed_model_name)
        try:
            self.local_model = SentenceTransformer(model_name, local_files_only=True)
        except TypeError:
            self.local_model = SentenceTransformer(model_name)
        return self.local_model

    def _resolve_local_model_path(self, model_name: str) -> str:
        path = Path(model_name).expanduser()
        if not path.exists():
            return model_name
        if (path / "config.json").exists() or (path / "modules.json").exists():
            return path.as_posix()
        snapshots_dir = path / "snapshots"
        if snapshots_dir.is_dir():
            snapshots = sorted(
                [item for item in snapshots_dir.iterdir() if item.is_dir()],
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            )
            if snapshots:
                return snapshots[0].as_posix()
        return path.as_posix()

    def embed_text(self, text: str) -> List[float]:
        vectors = self.embed_texts([text])
        return vectors[0]

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        clean_texts = [text for text in texts if text and text.strip()]
        if not clean_texts:
            return []

        if self._uses_local_model():
            vectors = self._get_local_model().encode(clean_texts, normalize_embeddings=True).tolist()
            self._validate_dimensions(vectors)
            return vectors

        response = self._get_client().embeddings.create(
            input=clean_texts,
            model=self.config.embed_model_name,
            dimensions=self.config.embed_vector_size,
        )
        vectors = [item.embedding for item in response.data]
        self._validate_dimensions(vectors)
        return vectors

    def _validate_dimensions(self, vectors: List[List[float]]) -> None:
        for vector in vectors:
            if len(vector) != self.config.embed_vector_size:
                raise ValueError(
                    f"Embedding 维度不匹配：模型输出 {len(vector)} 维，"
                    f"但 EMBED_VECTOR_SIZE={self.config.embed_vector_size}。"
                )
