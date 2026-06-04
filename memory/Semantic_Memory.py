import re
from typing import List, Dict, Any, Optional

from config import AppConfig, get_config
from memory.content_filters import is_polluted_context


class SemanticMemory:
    def __init__(
        self,
        qdrant_client: Any,
        neo4j_client=None,
        collection_name: Optional[str] = None,
        config: Optional[AppConfig] = None,
    ):
        self.config = config or get_config()
        self.qdrant = qdrant_client
        self.collection_name = collection_name or self.config.semantic_collection_name
        self.neo4j = neo4j_client # 预留给知识图谱

    def retrieve_from_graph(
        self,
        query: str,
        candidate_content: str = "",
        candidate_payload: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        基于 Neo4j 的轻量图谱相关性打分，返回 0.0-1.0。

        当前规则：
        1. 从用户 query 与候选片段中抽取实体词。
        2. 查询图谱中是否存在 query 实体到候选实体的 1-2 跳路径。
        3. 直接相连给高分，2 跳路径给中等分；如果图谱不可用或无命中，返回 0。

        Neo4j 中推荐的最小结构：
        (:Entity {name: "SENet"})
        (:Entity {name: "SE block"})
        (:Entity {name: "通道注意力"})
        (:Entity {name: "特征重标定"})
        (:Entity)-[:RELATED_TO|:CONTRIBUTES_TO|:USES|:HAS_CONCEPT]->(:Entity)
        """
        if not self.neo4j:
            return 0.0

        if hasattr(self.neo4j, "retrieve_graph_score"):
            return self._clamp_score(
                self.neo4j.retrieve_graph_score(
                    query=query,
                    candidate_content=candidate_content,
                    candidate_payload=candidate_payload or {},
                )
            )

        query_entities = self._extract_entities(query)
        candidate_entities = self._extract_entities(candidate_content)
        payload_entities = self._payload_entities(candidate_payload or {})
        candidate_entities = self._dedupe_entities([*candidate_entities, *payload_entities])

        if not query_entities or not candidate_entities:
            return 0.0

        cypher = """
        UNWIND $query_entities AS query_entity
        UNWIND $candidate_entities AS candidate_entity
        MATCH (q:Entity)
        WHERE toLower(q.name) = toLower(query_entity)
           OR any(alias IN coalesce(q.aliases, []) WHERE toLower(alias) = toLower(query_entity))
        MATCH (c:Entity)
        WHERE toLower(c.name) = toLower(candidate_entity)
           OR any(alias IN coalesce(c.aliases, []) WHERE toLower(alias) = toLower(candidate_entity))
        OPTIONAL MATCH path = shortestPath((q)-[*1..2]-(c))
        WITH q, c, path
        WHERE path IS NOT NULL
        RETURN
            q.name AS query_entity,
            c.name AS candidate_entity,
            length(path) AS distance,
            [rel IN relationships(path) | type(rel)] AS relation_types
        ORDER BY distance ASC
        LIMIT 10
        """
        records = self._run_graph_query(cypher, {
            "query_entities": query_entities,
            "candidate_entities": candidate_entities,
        })
        return self._score_graph_records(records)

    def retrieve(self, query: str, query_vector: List[float], query_importance: float, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        向量与图谱混合检索
        注意：语义记忆的“重要性”通常来自于当前提问(Query)的重要性，而不是知识片段本身的重要性。
        """
        # 1. 纯向量检索 (Qdrant)
        search_results = self._vector_search(query_vector=query_vector, limit=top_k * 2)

        final_results = []
        
        # 2. 混合打分逻辑
        for hit in search_results:
            vector_sim = hit.score
            # 假设你的文档内容存在 Qdrant 的 payload 中
            content = hit.payload.get("page_content") or hit.payload.get("content", "")
            if is_polluted_context(content):
                continue
            
            graph_sim = self.retrieve_from_graph(query, candidate_content=content, candidate_payload=hit.payload)
            
            if self.neo4j:
                # Formula: (Vec_Sim * 0.7 + Graph_Sim * 0.3) * (0.8 + Query_Importance * 0.4)
                mixed_similarity = vector_sim * 0.7 + graph_sim * 0.3
            else:
                mixed_similarity = vector_sim
            final_score = mixed_similarity * (0.8 + query_importance * 0.4)
            
            final_results.append({
                "content": content,
                "score": final_score,
                "vector_sim": vector_sim,
                "graph_sim": graph_sim,
                "payload": hit.payload,
            })

        final_results.sort(key=lambda x: x["score"], reverse=True)
        return final_results[:top_k]

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

    def _extract_entities(self, text: str) -> List[str]:
        if not text:
            return []
        candidates = re.findall(r"[A-Za-z][A-Za-z0-9_.+-]{1,}|[\u4e00-\u9fff]{2,}", text)
        stopwords = {
            "请问", "什么", "如何", "为什么", "怎么", "核心", "贡献", "内容", "介绍",
            "说明", "解释", "相关", "用户", "问题", "回答", "文档", "片段",
        }
        return self._dedupe_entities([
            item.strip()
            for item in candidates
            if item.strip() and item.strip() not in stopwords
        ])

    def _payload_entities(self, payload: Dict[str, Any]) -> List[str]:
        entities = payload.get("entities") or payload.get("entity_names") or []
        if isinstance(entities, str):
            return self._extract_entities(entities)
        if isinstance(entities, list):
            return self._dedupe_entities([str(item).strip() for item in entities if str(item).strip()])
        return []

    def _dedupe_entities(self, entities: List[str]) -> List[str]:
        deduped = []
        seen = set()
        for entity in entities:
            key = entity.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(entity)
        return deduped

    def _run_graph_query(self, cypher: str, parameters: Dict[str, Any]) -> List[Any]:
        try:
            if hasattr(self.neo4j, "query"):
                return list(self.neo4j.query(cypher, parameters) or [])
            if hasattr(self.neo4j, "run"):
                return list(self.neo4j.run(cypher, parameters) or [])
            if hasattr(self.neo4j, "execute_query"):
                result = self.neo4j.execute_query(cypher, parameters_=parameters)
                if isinstance(result, tuple):
                    return list(result[0] or [])
                return list(result or [])
            if hasattr(self.neo4j, "session"):
                database = getattr(self.config, "neo4j_database", None) or None
                session_kwargs = {"database": database} if database else {}
                with self.neo4j.session(**session_kwargs) as session:
                    return list(session.run(cypher, parameters))
        except Exception:
            return []
        return []

    def _score_graph_records(self, records: List[Any]) -> float:
        if not records:
            return 0.0

        best = 0.0
        for record in records:
            distance = self._record_get(record, "distance", 99)
            relation_types = self._record_get(record, "relation_types", []) or []
            if distance == 0:
                base = 1.0
            elif distance == 1:
                base = 0.95
            elif distance == 2:
                base = 0.65
            else:
                base = 0.2

            relation_bonus = 0.05 if relation_types else 0.0
            best = max(best, base + relation_bonus)
        return self._clamp_score(best)

    def _record_get(self, record: Any, key: str, default: Any = None) -> Any:
        if isinstance(record, dict):
            return record.get(key, default)
        if hasattr(record, "get"):
            try:
                return record.get(key, default)
            except TypeError:
                pass
        try:
            return record[key]
        except Exception:
            return default

    def _clamp_score(self, score: Any) -> float:
        try:
            value = float(score)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, value))
