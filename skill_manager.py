import importlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from config import AppConfig, get_config


@dataclass
class SkillResult:
    name: str
    content: str
    success: bool = True
    metadata: Dict[str, Any] = None

    def to_memory(self) -> Dict[str, Any]:
        return {
            "role": f"skill:{self.name}",
            "type": "skill",
            "content": self.content,
            "metadata": self.metadata or {},
        }


class Skill(Protocol):
    name: str
    description: str
    trigger_keywords: List[str]

    def should_activate(self, question: str) -> bool:
        ...

    def run(self, question: str, context: Optional[Dict[str, Any]] = None) -> SkillResult:
        ...


class SkillManager:
    """Load local Python skills and expose their output as QA context."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        skills: Optional[List[Skill]] = None,
    ):
        self.config = config or get_config()
        self.skills = skills if skills is not None else self._load_configured_skills()

    def retrieve_context(self, question: str) -> List[Dict[str, Any]]:
        memories = []
        for skill in self.skills:
            if not self._should_activate(skill, question):
                continue
            result = skill.run(question, context={"config": self.config})
            if result and result.content:
                memories.append(result.to_memory())
        return memories

    def list_skills(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": getattr(skill, "name", skill.__class__.__name__),
                "description": getattr(skill, "description", ""),
                "trigger_keywords": getattr(skill, "trigger_keywords", []),
            }
            for skill in self.skills
        ]

    def _load_configured_skills(self) -> List[Skill]:
        skills = []
        for name in self._configured_skill_names():
            module_path = f"{self.config.skills_package}.{name}.skill"
            module = importlib.import_module(module_path)
            skill = getattr(module, "SKILL", None)
            if skill is None and hasattr(module, "create_skill"):
                skill = module.create_skill()
            if skill is None:
                raise ValueError(f"Skill 模块 {module_path} 未暴露 SKILL 或 create_skill()。")
            skills.append(skill)
        return skills

    def _configured_skill_names(self) -> List[str]:
        return [
            item.strip()
            for item in self.config.skills_enabled_names.split(",")
            if item.strip()
        ]

    def _should_activate(self, skill: Skill, question: str) -> bool:
        lowered = (question or "").lower()
        compact_question = "".join(ch for ch in lowered if ch.isalnum())
        explicit_names = [
            getattr(skill, "name", ""),
            skill.__class__.__name__,
        ]
        for name in explicit_names:
            normalized = "".join(ch for ch in str(name).lower() if ch.isalnum())
            if normalized and normalized in compact_question:
                return True
        if hasattr(skill, "should_activate"):
            return bool(skill.should_activate(question))
        keywords = getattr(skill, "trigger_keywords", [])
        return any(keyword.lower() in lowered for keyword in keywords)
