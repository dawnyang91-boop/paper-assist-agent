import sqlite3
import time
import math
from types import SimpleNamespace
from typing import List, Dict, Any, Optional

from config import AppConfig, get_config
from memory.importance_scorer import clamp_importance, score_memory_importance
from rag.qdrant_utils import ensure_collection_vector_size

try:
    from qdrant_client.http import models
except ImportError:
    models = None


class EpisodicMemory:
    def __init__(
        self,
        qdrant_client: Any,
        db_path: Optional[str] = None,
        collection_name: Optional[str] = None,
        config: Optional[AppConfig] = None,
    ):
        self.config = config or get_config()
        self.qdrant = qdrant_client
        self.collection_name = collection_name or self.config.episodic_collection_name
        self.sqlite_conn = sqlite3.connect(db_path or self.config.episodic_db_path, check_same_thread=False)
        self._init_sqlite()
        # 注意：此处需确保 Qdrant 中已有对应的 collection

    def _init_sqlite(self):
        """初始化 SQLite 结构化数据库"""
        cursor = self.sqlite_conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_meta (
                memory_id TEXT PRIMARY KEY,
                session_id TEXT,
                timestamp REAL,
                importance_score REAL,
                content TEXT
            )
        ''')
        self.sqlite_conn.commit()

    def calculate_recency(self, past_timestamp: float, decay_rate: Optional[float] = None) -> float:
        """
        计算时间近因性 (指数衰减)。
        decay_rate 控制衰减速度，可根据实际按小时/天调整。
        """
        decay_rate = self.config.memory_decay_rate if decay_rate is None else decay_rate
        time_diff = (time.time() - past_timestamp) / 3600 # 转换为小时差
        return math.exp(-decay_rate * time_diff)

    def retrieve(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        混合检索核心逻辑
        """
        # 1. 从 Qdrant 进行初步的向量相似度检索 (召回更多候选集以便二次打分)
        search_results = self._vector_search(
            query_vector=query_vector,
            limit=top_k * 3,  # 扩大候选集
        )

        final_results = []
        cursor = self.sqlite_conn.cursor()

        # 2. 结合 SQLite 中的元数据，套用思维导图中的公式进行重排
        for hit in search_results:
            memory_id = hit.id
            vector_sim = hit.score  # 假设 Qdrant 使用的是 Cosine 相似度，范围在0-1左右
            
            # 查询 SQLite 获取元数据
            cursor.execute("SELECT timestamp, importance_score, content FROM memory_meta WHERE memory_id=?", (memory_id,))
            row = cursor.fetchone()
            if not row:
                continue
                
            timestamp, importance, content = row
            recency = self.calculate_recency(timestamp)
            
            # 核心打分公式实现
            # Formula: (Sim * 0.8 + Recency * 0.2) * (0.8 + Importance * 0.4)
            final_score = (vector_sim * 0.8 + recency * 0.2) * (0.8 + importance * 0.4)
            
            final_results.append({
                "content": content,
                "score": final_score,
                "vector_sim": vector_sim,
                "importance": importance
            })

        # 3. 根据最终得分降序排序并截断
        final_results.sort(key=lambda x: x["score"], reverse=True)
        return final_results[:top_k]
        
    def add_memory(
        self,
        memory_id: str,
        session_id: str,
        content: str,
        vector: List[float],
        importance_score: Optional[float] = None,
        use_llm_importance: bool = False,
        role: Optional[str] = None,
        context: Optional[str] = None,
    ):
        """存入新记忆（此方法通常在工作记忆溢出时调用）"""
        if importance_score is None:
            importance_result = score_memory_importance(
                content=content,
                role=role,
                memory_type="episodic",
                context=context,
                use_llm=use_llm_importance,
                config=self.config,
            )
            importance_score = importance_result.score
        else:
            importance_score = clamp_importance(importance_score)

        # 1. 写入 SQLite
        cursor = self.sqlite_conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO memory_meta VALUES (?, ?, ?, ?, ?)",
            (memory_id, session_id, time.time(), importance_score, content)
        )
        self.sqlite_conn.commit()
        
        # 2. 写入 Qdrant
        ensure_collection_vector_size(self.qdrant, self.collection_name, len(vector))
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=[self._make_point(memory_id, vector, {
                "session_id": session_id,
                "content": content,
                "importance": importance_score,
                "timestamp": time.time(),
            })]
        )

    def _make_point(self, memory_id: str, vector: List[float], payload: Dict[str, Any]):
        if models is not None:
            return models.PointStruct(id=memory_id, vector=vector, payload=payload)
        return SimpleNamespace(id=memory_id, vector=vector, payload=payload)

    def _vector_search(self, query_vector: List[float], limit: int):
        if hasattr(self.qdrant, "search"):
            return self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
            )
        if hasattr(self.qdrant, "query_points"):
            response = self.qdrant.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
            )
            return getattr(response, "points", response)
        raise AttributeError(
            "当前 QdrantClient 既没有 search，也没有 query_points，"
            "请检查 qdrant-client 版本或传入兼容的客户端。"
        )
