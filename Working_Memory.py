import time
import math
from collections import deque, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional

from config import AppConfig, get_config
from importance_scorer import clamp_importance, score_memory_importance
from redis_runtime import RedisUnavailable, get_redis_runtime
from redis_working_memory import RedisWorkingMemoryStore


class WorkingMemory:
    def __init__(
        self,
        max_capacity: Optional[int] = None,
        ttl_seconds: Optional[int] = None,
        config: Optional[AppConfig] = None,
    ):
        self.config = config or get_config()
        max_capacity = max_capacity or self.config.working_memory_max_capacity
        ttl_seconds = ttl_seconds or self.config.working_memory_ttl_seconds
        self.max_capacity = max_capacity
        self.ttl_seconds = ttl_seconds
        
        # 核心改造：使用 defaultdict，为每个独立的 session_id 维护一个专属的 deque
        self.sessions = defaultdict(lambda: deque(maxlen=max_capacity))
        
        # 溢出缓冲区（可以全局共享，因为每条数据里都打上了 session_id 标签）
        self.overflow_buffer = []
        self.redis_store = None
        try:
            runtime = get_redis_runtime(self.config)
            if runtime is not None:
                self.redis_store = RedisWorkingMemoryStore(runtime=runtime, config=self.config)
        except RedisUnavailable as exc:
            print(f"[警告] Redis 工作记忆不可用，已回退到进程内存：{exc}")

    def add_memory(self, session_id: str, role: str, content: str, importance: Optional[float] = None):
        """添加新对话，精确路由到对应的 session_id 队列中"""
        session_buffer = self.sessions[session_id]
        if importance is None:
            importance_result = score_memory_importance(
                content=content,
                role=role,
                memory_type="working",
                use_llm=False,
                config=self.config,
            )
            importance = importance_result.score
        else:
            importance = clamp_importance(importance)
        
        # 1. 拦截逻辑：检查该 session 的队列是否已满
        if len(session_buffer) == self.max_capacity:
            evicted_memory = session_buffer.popleft()
            self.overflow_buffer.append(evicted_memory)
            
        # 2. 加入新记忆
        new_memory = {
            "session_id": session_id,
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "importance": importance
        }
        session_buffer.append(new_memory)
        if self.redis_store is not None:
            self.redis_store.append(session_id, new_memory)

    def _cleanup_expired(self, session_id: str):
        """仅针对当前 session 进行 TTL 自动清理"""
        if session_id not in self.sessions:
            return
            
        current_time = time.time()
        session_buffer = self.sessions[session_id]
        
        # 筛选未过期的数据
        active_memories = [
            m for m in session_buffer 
            if (current_time - m["timestamp"]) < self.ttl_seconds
        ]
        
        # 更新该 session 的队列
        self.sessions[session_id] = deque(active_memories, maxlen=self.max_capacity)

    def retrieve(self, query: str, session_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """基于特定 session_id 的工作记忆检索"""
        # 1. 清理该 session 下的过期数据
        self._cleanup_expired(session_id)
        
        session_buffer = self.redis_store.read_recent(session_id, self.max_capacity) if self.redis_store is not None else self.sessions.get(session_id)
        
        # 如果该 session 为空，直接返回
        if not session_buffer:
            return []

        # 2. 仅提取当前 session 的文本用于构建 TF-IDF
        corpus = [m["content"] for m in session_buffer]
        corpus.append(query)

        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform(corpus)
            cosine_sims = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1]).flatten()
        except ValueError:
            cosine_sims = [0.0] * len(session_buffer)

        final_results = []
        for idx, memory in enumerate(session_buffer):
            sim_score = cosine_sims[idx]
            time_diff = (time.time() - memory["timestamp"]) / 3600
            time_decay = math.exp(-0.01 * time_diff)
            importance = memory["importance"]
            
            final_score = (sim_score * time_decay) * (0.8 + importance * 0.4)
            
            final_results.append({
                "session_id": memory["session_id"],
                "role": memory["role"],
                "content": memory["content"],
                "timestamp": memory["timestamp"],
                "importance": importance,
                "score": final_score
            })

        final_results.sort(key=lambda x: x["score"], reverse=True)
        return final_results[:top_k]
        
    def get_overflow_memories(self, session_id: str = None) -> List[Dict[str, Any]]:
        """
        获取溢出的记忆。
        支持指定 session_id 获取，或者不传参数获取所有 session 的溢出记忆。
        """
        if not self.overflow_buffer:
            return []
            
        if session_id:
            # 仅提取特定 session 的溢出数据
            session_overflows = [m for m in self.overflow_buffer if m["session_id"] == session_id]
            # 从主缓冲区中移除已提取的数据
            self.overflow_buffer = [m for m in self.overflow_buffer if m["session_id"] != session_id]
            return session_overflows
        else:
            # 提取全部并清空
            all_overflows = list(self.overflow_buffer)
            self.overflow_buffer.clear()
            return all_overflows

    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """获取指定 session 的当前工作记忆。"""
        self._cleanup_expired(session_id)
        if self.redis_store is not None:
            return self.redis_store.read_recent(session_id, self.max_capacity)
        return list(self.sessions.get(session_id, []))

    def clear_session(self, session_id: str):
        """清空指定 session 的工作记忆。"""
        self.sessions.pop(session_id, None)
        if self.redis_store is not None:
            self.redis_store.clear(session_id)
