import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

from config import AppConfig, get_config
from content_filters import is_polluted_context
from embedding_service import EmbeddingService
from importance_scorer import clamp_importance, score_memory_importance, score_overflow_memory
from qdrant_utils import ensure_collection_vector_size
from redis_runtime import RedisUnavailable, get_redis_runtime
from redis_task_queue import RedisTaskStore


try:
    from qdrant_client.http import models
except ImportError:
    models = None


@dataclass
class ConsolidationResult:
    session_id: str
    overflow_count: int
    episodic_memory_id: Optional[str]
    summary: str
    importance: int


@dataclass
class FactWriteResult:
    facts: List[str]
    semantic_ids: List[str] = field(default_factory=list)
    graph_written: bool = False
    errors: List[str] = field(default_factory=list)


@dataclass
class SensoryWriteResult:
    memory_id: str
    modality: str
    caption: str
    source_path: str
    importance: int


class MemoryManager:
    """Coordinate working, episodic, semantic, and sensory memory modules."""

    def __init__(
        self,
        working_memory: Any,
        episodic_memory: Any = None,
        semantic_memory: Any = None,
        sensory_memory: Any = None,
        embedding_service: Optional[EmbeddingService] = None,
        config: Optional[AppConfig] = None,
        llm_client: Any = None,
        audio_transcriber: Any = None,
    ):
        self.config = config or get_config()
        self.working_memory = working_memory
        self.episodic_memory = episodic_memory
        self.semantic_memory = semantic_memory
        self.sensory_memory = sensory_memory
        self.embedding_service = embedding_service or EmbeddingService(config=self.config)
        self.llm_client = llm_client
        self.audio_transcriber = audio_transcriber
        self.redis_task_store = None
        try:
            runtime = get_redis_runtime(self.config)
            if runtime is not None:
                self.redis_task_store = RedisTaskStore(runtime=runtime, config=self.config)
        except RedisUnavailable as exc:
            print(f"[警告] Redis 任务队列不可用，记忆压缩将使用同步路径：{exc}")

    def add_interaction(
        self,
        session_id: str,
        user_message: str,
        assistant_answer: str,
        user_importance: Optional[float] = None,
        assistant_importance: Optional[float] = None,
        consolidate_overflow: bool = True,
    ) -> List[ConsolidationResult]:
        """Add one QA turn to working memory, then consolidate any overflow."""
        self.working_memory.add_memory(session_id, "user", user_message, importance=user_importance)
        self.working_memory.add_memory(session_id, "assistant", assistant_answer, importance=assistant_importance)
        if not consolidate_overflow:
            return []
        result = self.consolidate_working_overflow(session_id)
        return [result] if result else []

    def consolidate_working_overflow(self, session_id: Optional[str] = None) -> Optional[ConsolidationResult]:
        """Summarize overflowed working memories and persist them into episodic memory."""
        if self.episodic_memory is None:
            return None

        overflow = self.working_memory.get_overflow_memories(session_id)
        if not overflow:
            return None

        effective_session_id = session_id or overflow[0].get("session_id", "unknown-session")
        task_id = None
        if self.redis_task_store is not None:
            task_id = self.redis_task_store.create_task(
                "memory_overflow_compress",
                session_id=effective_session_id,
                input_data={"overflow_count": len(overflow)},
            )
            self.redis_task_store.enqueue_overflow(task_id, effective_session_id, overflow)
            self.redis_task_store.update_task(task_id, {"status": "running", "progress": 20, "message": "正在压缩工作记忆"})
        summary = self.summarize_memories(overflow)
        importance = score_overflow_memory(
            {"role": "summary", "content": summary},
            context=f"session_id={effective_session_id}",
            use_llm=True,
            config=self.config,
            client=self.llm_client,
        ).score
        vector = self.embedding_service.embed_text(summary)
        memory_id = self._stable_id("episodic", effective_session_id, summary)

        self.episodic_memory.add_memory(
            memory_id=memory_id,
            session_id=effective_session_id,
            content=summary,
            vector=vector,
            importance_score=importance,
            role="summary",
            context="working_memory_overflow",
        )
        if self.redis_task_store is not None and task_id:
            self.redis_task_store.update_task(
                task_id,
                {
                    "status": "succeeded",
                    "progress": 100,
                    "message": "工作记忆压缩完成",
                    "result_json": {
                        "episodic_memory_id": memory_id,
                        "importance": importance,
                        "summary": summary,
                    },
                    "finished_at": time.time(),
                },
            )
        return ConsolidationResult(
            session_id=effective_session_id,
            overflow_count=len(overflow),
            episodic_memory_id=memory_id,
            summary=summary,
            importance=importance,
        )

    def summarize_memories(self, memories: Iterable[Dict[str, Any]]) -> str:
        memories_list = list(memories)
        if not memories_list:
            return ""

        llm_summary = self._llm_summarize_memories(memories_list)
        if llm_summary:
            return llm_summary

        lines = []
        for memory in memories_list:
            role = memory.get("role", "memory")
            content = memory.get("content", "")
            if content:
                lines.append(f"{role}: {content}")
        summary = "\n".join(lines)
        max_chars = self.config.memory_summary_max_chars
        if len(summary) > max_chars:
            summary = summary[: max_chars - 3] + "..."
        return summary

    def process_high_value_qa(
        self,
        question: str,
        answer: str,
        session_id: Optional[str] = None,
        min_importance: Optional[int] = None,
    ) -> FactWriteResult:
        """Extract high-value facts from a QA pair and persist them into semantic memory/Neo4j."""
        if is_polluted_context(answer):
            return FactWriteResult(facts=[], semantic_ids=[], graph_written=False, errors=["跳过污染型或拒答型回答。"])
        min_importance = min_importance or self.config.memory_semantic_fact_min_importance
        facts = self.extract_facts_from_qa(question, answer)
        accepted = []
        semantic_ids = []
        errors = []
        for fact in facts:
            importance = score_memory_importance(
                content=fact,
                role="fact",
                memory_type="semantic",
                context=f"question={question}",
                use_llm=False,
                config=self.config,
            ).score
            if importance < min_importance:
                continue
            accepted.append(fact)
            try:
                semantic_id = self.write_semantic_fact(fact, session_id=session_id, importance=importance)
            except Exception as exc:
                errors.append(f"写入语义记忆失败：{exc}")
                semantic_id = None
            if semantic_id:
                semantic_ids.append(semantic_id)

        try:
            graph_written = self.write_facts_to_graph(accepted, session_id=session_id)
        except Exception as exc:
            errors.append(f"写入 Neo4j 图谱失败：{exc}")
            graph_written = False
        return FactWriteResult(facts=accepted, semantic_ids=semantic_ids, graph_written=graph_written, errors=errors)

    def extract_facts_from_qa(self, question: str, answer: str) -> List[str]:
        llm_facts = self._llm_extract_facts(question, answer)
        if llm_facts:
            return llm_facts

        text = f"问题：{question}\n回答：{answer}"
        importance = score_memory_importance(
            content=text,
            role="qa",
            memory_type="semantic",
            use_llm=False,
            config=self.config,
        ).score
        if importance >= self.config.memory_semantic_fact_min_importance:
            return [answer.strip()]
        return []

    def write_semantic_fact(
        self,
        fact: str,
        session_id: Optional[str] = None,
        importance: int = 7,
    ) -> Optional[str]:
        if self.semantic_memory is None or not fact.strip():
            return None

        if hasattr(self.semantic_memory, "add_fact"):
            return self.semantic_memory.add_fact(fact, session_id=session_id, importance=importance)

        qdrant = getattr(self.semantic_memory, "qdrant", None)
        collection_name = getattr(self.semantic_memory, "collection_name", self.config.semantic_collection_name)
        if qdrant is None:
            return None

        vector = self.embedding_service.embed_text(fact)
        semantic_id = self._stable_id("semantic", session_id or "global", fact)
        payload = {
            "page_content": fact,
            "content": fact,
            "memory_type": "semantic_fact",
            "session_id": session_id,
            "importance": clamp_importance(importance),
            "content_hash": self._content_hash(fact),
            "created_at": time.time(),
        }
        self._ensure_vector_collection(qdrant, collection_name, vector_size=len(vector))
        qdrant.upsert(
            collection_name=collection_name,
            points=[self._make_point(semantic_id, vector, payload)],
        )
        return semantic_id

    def _ensure_vector_collection(self, qdrant, collection_name: str, vector_size: int) -> None:
        ensure_collection_vector_size(qdrant, collection_name, vector_size)

    def write_facts_to_graph(self, facts: List[str], session_id: Optional[str] = None) -> bool:
        if not facts or self.semantic_memory is None:
            return False
        neo4j = getattr(self.semantic_memory, "neo4j", None)
        if neo4j is None:
            return False

        try:
            if hasattr(neo4j, "write_facts"):
                neo4j.write_facts(facts, session_id=session_id)
                return True
            if hasattr(neo4j, "add_facts"):
                neo4j.add_facts(facts, session_id=session_id)
                return True
        except Exception:
            return False
        return False

    def process_media_input(
        self,
        session_id: str,
        file_path: str,
        modality: Optional[str] = None,
        importance: Optional[float] = None,
    ) -> SensoryWriteResult:
        """Caption and embed image/audio input, then write it to sensory memory."""
        if self.sensory_memory is None:
            raise ValueError("未配置 sensory_memory，无法写入感知记忆。")

        modality = modality or self.detect_modality(file_path)
        caption = self.generate_caption(file_path, modality=modality)
        importance_score = clamp_importance(
            importance if importance is not None else score_memory_importance(
                content=caption,
                role="media",
                memory_type="sensory",
                use_llm=False,
                config=self.config,
            ).score
        )

        if modality == "image" and hasattr(self.sensory_memory, "get_image_embedding"):
            try:
                vector = self.sensory_memory.get_image_embedding(file_path)
            except RuntimeError:
                vector = self.embedding_service.embed_text(caption)
        else:
            vector = self.embedding_service.embed_text(caption)

        memory_id = self._stable_id("sensory", session_id, f"{file_path}:{caption}")
        payload = {
            "session_id": session_id,
            "modality": modality,
            "source_path": file_path,
            "caption": caption,
            "content": caption,
            "timestamp": time.time(),
            "importance": importance_score,
        }

        if hasattr(self.sensory_memory, "add_memory"):
            self.sensory_memory.add_memory(memory_id=memory_id, vector=vector, payload=payload)
        else:
            qdrant = getattr(self.sensory_memory, "qdrant", None)
            collection_name = getattr(self.sensory_memory, "collection_name", self.config.sensory_collection_name)
            if qdrant is None:
                raise ValueError("sensory_memory 缺少 qdrant 客户端，无法写入感知记忆。")
            qdrant.upsert(
                collection_name=collection_name,
                points=[self._make_point(memory_id, vector, payload)],
            )

        return SensoryWriteResult(
            memory_id=memory_id,
            modality=modality,
            caption=caption,
            source_path=file_path,
            importance=importance_score,
        )

    def generate_caption(self, file_path: str, modality: str) -> str:
        if modality == "audio":
            transcript = self.transcribe_audio(file_path)
            return f"音频转写：{transcript}"

        llm_caption = self._llm_caption(file_path, modality)
        if llm_caption:
            return llm_caption
        return f"{modality} 文件：{os.path.basename(file_path)}"

    def transcribe_audio(self, file_path: str) -> str:
        if self.audio_transcriber is None:
            return f"{os.path.basename(file_path)}（未配置 ASR，仅记录文件名）"
        if callable(self.audio_transcriber):
            return str(self.audio_transcriber(file_path))
        if hasattr(self.audio_transcriber, "transcribe"):
            return str(self.audio_transcriber.transcribe(file_path))
        return f"{os.path.basename(file_path)}（ASR 接口不可用）"

    def retrieve_memories(
        self,
        query: str,
        session_id: Optional[str] = None,
        query_vector: Optional[List[float]] = None,
        top_k: int = 5,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Retrieve from all configured memories with one call."""
        query_vector = query_vector or self.embedding_service.embed_text(query)
        results = {}
        diagnostics = {}

        if self.working_memory is not None and session_id:
            try:
                results["working"] = self.working_memory.retrieve(query, session_id=session_id, top_k=top_k)
                diagnostics["working"] = {"count": len(results["working"])}
            except Exception as exc:
                results["working"] = []
                diagnostics["working"] = {"count": 0, "error": str(exc)}
        if self.episodic_memory is not None:
            try:
                results["episodic"] = self.episodic_memory.retrieve(query_vector=query_vector, top_k=top_k)
                diagnostics["episodic"] = {"count": len(results["episodic"])}
            except Exception as exc:
                results["episodic"] = []
                diagnostics["episodic"] = {"count": 0, "error": str(exc)}
        if self.semantic_memory is not None:
            try:
                results["semantic"] = self.semantic_memory.retrieve(
                    query=query,
                    query_vector=query_vector,
                    query_importance=5.0,
                    top_k=top_k,
                )
                diagnostics["semantic"] = {"count": len(results["semantic"])}
            except Exception as exc:
                results["semantic"] = []
                diagnostics["semantic"] = {"count": 0, "error": str(exc)}
        if self.sensory_memory is not None:
            results["sensory"], diagnostics["sensory"] = self._retrieve_sensory_memory(
                query=query,
                query_vector=query_vector,
                top_k=top_k,
            )
        results["_diagnostics"] = diagnostics
        return results

    def _retrieve_sensory_memory(
        self,
        query: str,
        query_vector: List[float],
        top_k: int,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        try:
            results = self.sensory_memory.retrieve(query_vector=query_vector, top_k=top_k)
            return results, {"count": len(results), "query_vector_dim": len(query_vector)}
        except Exception as exc:
            first_error = str(exc)
            if "Vector dimension error" not in first_error or not hasattr(self.sensory_memory, "get_text_embedding"):
                return [], {"count": 0, "error": first_error, "query_vector_dim": len(query_vector)}

        try:
            sensory_query_vector = self.sensory_memory.get_text_embedding(query)
            results = self.sensory_memory.retrieve(query_vector=sensory_query_vector, top_k=top_k)
            return results, {
                "count": len(results),
                "query_vector_dim": len(sensory_query_vector),
                "fallback": "sensory_text_embedding",
                "first_error": first_error,
            }
        except Exception as exc:
            return [], {
                "count": 0,
                "skipped": True,
                "reason": (
                    "感知记忆 collection 与通用文本 embedding 维度不一致，且当前未加载可用的感知文本编码器。"
                    "如需在 Web 中检索图片/音频感知记忆，请用 load_sensory_model=True 启动或重建 sensory collection。"
                ),
                "query_vector_dim": len(query_vector),
                "first_error": first_error,
                "fallback_error": str(exc),
            }

    def detect_modality(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}:
            return "image"
        if ext in {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg"}:
            return "audio"
        return "text"

    def _llm_summarize_memories(self, memories: List[Dict[str, Any]]) -> str:
        client = self._get_llm_client()
        if client is None:
            return ""
        lines = [f"{item.get('role', 'memory')}: {item.get('content', '')}" for item in memories]
        prompt = "Compress the following overflowed working memories into one episodic memory. Preserve key facts, user preferences, task decisions, and todos. Output Chinese summary text only:\n" + "\n".join(lines)
        try:
            response = client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": "You compress conversation memories. Output only the summary body in Chinese."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=self.config.memory_summary_max_tokens,
                **self.config.chat_completion_kwargs(),
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return ""

    def _llm_extract_facts(self, question: str, answer: str) -> List[str]:
        client = self._get_llm_client()
        if client is None:
            return []
        prompt = f"""Extract stable facts, user preferences, project decisions, or key constraints worth long-term storage from the QA pair below.
Return only a JSON array. Return [] if there are no high-value facts. Write extracted facts in Chinese.

Question: {question}
Answer: {answer}"""
        try:
            response = client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": "Return a valid JSON array only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                **self.config.chat_completion_kwargs(),
            )
            data = self._parse_json(response.choices[0].message.content)
            if isinstance(data, dict):
                data = data.get("facts", [])
            return [str(item).strip() for item in data if str(item).strip()]
        except Exception:
            return []

    def _llm_caption(self, file_path: str, modality: str) -> str:
        client = self._get_llm_client()
        if client is None:
            return ""
        prompt = f"Generate one Chinese caption for this {modality} input so it can be retrieved as sensory memory. File name: {os.path.basename(file_path)}"
        try:
            response = client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": "Output exactly one Chinese caption."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=120,
                **self.config.chat_completion_kwargs(),
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return ""

    def _get_llm_client(self):
        if self.llm_client is not None:
            return self.llm_client
        if not self.config.openai_api_key:
            return None
        try:
            from openai import OpenAI
        except ImportError:
            return None
        self.llm_client = OpenAI(api_key=self.config.openai_api_key, base_url=self.config.openai_base_url)
        return self.llm_client

    def _parse_json(self, text: str):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start_candidates = [index for index in [text.find("["), text.find("{")] if index >= 0]
            if not start_candidates:
                raise
            start = min(start_candidates)
            end = max(text.rfind("]"), text.rfind("}"))
            return json.loads(text[start:end + 1])

    def _make_point(self, point_id: str, vector: List[float], payload: Dict[str, Any]):
        if models is not None:
            return models.PointStruct(id=point_id, vector=vector, payload=payload)
        return SimpleNamespace(id=point_id, vector=vector, payload=payload)

    def _stable_id(self, prefix: str, scope: str, content: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{prefix}:{scope}:{self._content_hash(content)}"))

    def _content_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
