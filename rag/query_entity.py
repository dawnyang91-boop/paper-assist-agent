import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List


GENERIC_ENTITY_TERMS = {
    "ai",
    "agent",
    "agents",
    "attack",
    "attacks",
    "llm",
    "model",
    "models",
    "rag",
    "security",
}


@dataclass
class QueryEntityProfile:
    entities: List[str] = field(default_factory=list)
    has_strong_entity: bool = False
    entity_types: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "entities": list(self.entities),
            "has_strong_entity": self.has_strong_entity,
            "entity_types": dict(self.entity_types),
        }


@dataclass
class EntityCoverage:
    required_entities: List[str] = field(default_factory=list)
    covered_entities: List[str] = field(default_factory=list)
    missing_entities: List[str] = field(default_factory=list)
    coverage_ratio: float = 1.0
    entity_coverage_failed: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "required_entities": list(self.required_entities),
            "covered_entities": list(self.covered_entities),
            "missing_entities": list(self.missing_entities),
            "coverage_ratio": self.coverage_ratio,
            "entity_coverage_failed": self.entity_coverage_failed,
        }


class QueryEntityExtractor:
    """Rule-based strong entity extractor for model names, paper names, acronyms, and IDs."""

    UPPER_RE = re.compile(r"\b[A-Z][A-Z0-9_-]{2,}\b")
    CAMEL_RE = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][A-Za-z0-9]+)+(?:[-_][A-Za-z0-9]+)?\b")
    MODEL_ID_RE = re.compile(r"\b[A-Za-z]+(?:-[A-Za-z0-9]+)+\b")
    QUOTED_RE = re.compile(r"[\"'“”‘’]([^\"'“”‘’]{2,60})[\"'“”‘’]")
    CN_HINT_RE = re.compile(r"(?:模型|方法|框架|论文|算法)\s*([A-Za-z][A-Za-z0-9_.+-]{2,})")

    def extract(self, question: str) -> QueryEntityProfile:
        text = question or ""
        typed: Dict[str, str] = {}
        candidates: List[tuple[str, str]] = []
        candidates.extend((item, "acronym") for item in self.UPPER_RE.findall(text))
        candidates.extend((item, "model_name") for item in self.CAMEL_RE.findall(text))
        candidates.extend((item, "model_id") for item in self.MODEL_ID_RE.findall(text))
        candidates.extend((item, "quoted") for item in self.QUOTED_RE.findall(text))
        candidates.extend((item, "cn_hint") for item in self.CN_HINT_RE.findall(text))

        entities = []
        seen = set()
        for raw, entity_type in candidates:
            entity = self._clean(raw)
            if not entity:
                continue
            key = entity.lower()
            if key in seen or self._is_generic(entity):
                continue
            seen.add(key)
            entities.append(entity)
            typed[entity] = entity_type

        return QueryEntityProfile(
            entities=entities,
            has_strong_entity=bool(entities),
            entity_types=typed,
        )

    def coverage(self, question: str, chunks: Iterable[object]) -> EntityCoverage:
        profile = self.extract(question)
        required = list(profile.entities)
        if not required:
            return EntityCoverage()

        haystack_parts = []
        for chunk in chunks:
            content = getattr(chunk, "content", "") or ""
            payload = getattr(chunk, "payload", {}) or {}
            haystack_parts.append(content)
            haystack_parts.extend(str(value) for value in payload.values() if value is not None)
        haystack = "\n".join(haystack_parts).lower()

        covered = [entity for entity in required if entity.lower() in haystack]
        missing = [entity for entity in required if entity not in covered]
        ratio = len(covered) / len(required) if required else 1.0
        return EntityCoverage(
            required_entities=required,
            covered_entities=covered,
            missing_entities=missing,
            coverage_ratio=round(ratio, 4),
            entity_coverage_failed=bool(required and not covered),
        )

    def _clean(self, value: str) -> str:
        return re.sub(r"^[^\w]+|[^\w.+-]+$", "", (value or "").strip())

    def _is_generic(self, value: str) -> bool:
        text = value.strip().lower()
        return text in GENERIC_ENTITY_TERMS
