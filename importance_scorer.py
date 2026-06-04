import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from config import AppConfig, get_config


IMPORTANCE_SCALE = """0-2: no long-term value, greetings, duplicate content
3-4: ordinary context with short-term usefulness
5-6: related to the current task and may be useful later
7-8: user preferences, project decisions, key facts
9-10: stable identity information, core goals, hard constraints"""


SCORING_PLAN = """Short-term working memory: use the default rule-based score unless a manual/programmatic importance is supplied.
Working-memory overflow -> episodic memory: use the LLM to judge importance.
Explicit user preferences, identity information, long-term goals, and key facts: high score.
Ordinary chat, duplicate content, and temporary context: low score.
Semantic knowledge-base documents: normally do not score every chunk with the LLM; rely on vector similarity and reranking first."""


@dataclass(frozen=True)
class ImportanceResult:
    score: int
    reason: str
    source: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "reason": self.reason,
            "source": self.source,
        }


def clamp_importance(score: Any) -> int:
    try:
        value = round(float(score))
    except (TypeError, ValueError):
        value = 5
    return max(0, min(10, value))


def rule_based_importance(
    content: str,
    role: Optional[str] = None,
    memory_type: str = "working",
    context: Optional[str] = None,
) -> ImportanceResult:
    """Lightweight fallback scorer for short-term memory and offline tests."""
    text = f"{context or ''}\n{content or ''}".strip()
    lower_text = text.lower()

    if not text:
        return ImportanceResult(0, "空内容没有长期记忆价值。", "rule")

    greeting_patterns = [
        "你好", "您好", "谢谢", "感谢", "早上好", "晚上好", "ok", "好的", "收到",
        "hello", "hi", "thanks", "thank you",
    ]
    if len(text) <= 20 and any(item in lower_text for item in greeting_patterns):
        return ImportanceResult(1, "偏寒暄或确认信息，长期价值很低。", "rule")

    strong_constraint_patterns = [
        "必须", "永远", "不要", "禁止", "一定要", "务必", "强约束", "原则",
        "以后都", "始终", "不能", "只允许",
    ]
    identity_goal_patterns = [
        "我的名字", "我叫", "我是", "我的身份", "我的目标", "长期目标",
        "核心目标", "职业", "公司", "项目目标", "系统目标",
    ]
    preference_patterns = [
        "我喜欢", "我不喜欢", "我偏好", "我希望", "请记住", "记住",
        "偏向", "习惯", "风格", "以后", "默认",
    ]
    decision_fact_patterns = [
        "决定", "选择", "采用", "使用", "架构", "方案", "规划", "collection",
        "模型", "数据库", "qdrant", "sqlite", "neo4j", "api key", "base_url",
        "python版本", "版本为", "已完成", "当前进度",
    ]
    task_patterns = [
        "请你", "帮我", "实现", "新增", "修复", "测试", "运行", "阅读",
        "完成", "项目", "代码", "文档", "问答系统", "rag", "memory",
    ]

    if any(item in lower_text for item in strong_constraint_patterns) and any(
        item in lower_text for item in identity_goal_patterns + preference_patterns
    ):
        return ImportanceResult(9, "包含长期身份/目标/偏好且带有强约束，应长期稳定保存。", "rule")

    if any(item in lower_text for item in identity_goal_patterns):
        return ImportanceResult(8, "包含用户身份、长期目标或系统核心目标。", "rule")

    if any(item in lower_text for item in preference_patterns):
        return ImportanceResult(7, "包含用户偏好、默认习惯或希望之后持续遵守的信息。", "rule")

    if any(item in lower_text for item in decision_fact_patterns):
        return ImportanceResult(7, "包含项目决策、技术事实或后续开发会复用的信息。", "rule")

    if any(item in lower_text for item in task_patterns):
        return ImportanceResult(5, "与当前任务相关，之后可能会作为短期上下文复用。", "rule")

    if role == "assistant" and len(text) > 80:
        return ImportanceResult(4, "助手回复有一定上下文价值，但未识别出长期事实。", "rule")

    if memory_type == "episodic":
        return ImportanceResult(4, "作为情景记忆有短期帮助，但长期价值不明确。", "rule")

    return ImportanceResult(3, "普通上下文，主要提供短期帮助。", "rule")


def build_importance_prompt(
    content: str,
    role: Optional[str] = None,
    memory_type: str = "episodic",
    context: Optional[str] = None,
) -> str:
    return f"""You are the memory-importance evaluator for an intelligent QA system. Score the candidate memory with an integer from 0 to 10 according to the plan and scale.

Scoring plan:
{SCORING_PLAN}

Score scale:
{IMPORTANCE_SCALE}

Evaluation principles:
- Evaluate only the value of this information for future QA and long-term personalization; do not judge writing quality.
- Explicit user preferences, identity information, long-term goals, key project decisions, stable facts, and hard constraints deserve high scores.
- Greetings, duplicate content, temporary process details, and information useful only for the current turn deserve low scores.
- Do not automatically give semantic knowledge-base chunks high scores just because they are technical; those are usually handled by vector search and reranking.
- The score must be an integer from 0 to 10.
- Return JSON only; do not output Markdown.

Candidate memory type: {memory_type}
Speaker role: {role or "unknown"}
Related context: {context or "none"}
Candidate memory content: {content}

Return:
{{"score": 0-10, "reason": "一句话说明原因"}}"""


def _parse_json_object(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def llm_importance(
    content: str,
    role: Optional[str] = None,
    memory_type: str = "episodic",
    context: Optional[str] = None,
    config: Optional[AppConfig] = None,
    client: Any = None,
) -> ImportanceResult:
    cfg = config or get_config()
    if client is None:
        if not cfg.openai_api_key:
            fallback = rule_based_importance(content, role=role, memory_type=memory_type, context=context)
            return ImportanceResult(fallback.score, f"未配置 OPENAI_API_KEY，使用规则兜底：{fallback.reason}", "rule")

        from openai import OpenAI

        client = OpenAI(api_key=cfg.openai_api_key, base_url=cfg.openai_base_url)

    prompt = build_importance_prompt(
        content=content,
        role=role,
        memory_type=memory_type,
        context=context,
    )
    try:
        response = client.chat.completions.create(
            model=cfg.model_name,
            messages=[
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            **cfg.chat_completion_kwargs(),
        )
        raw = response.choices[0].message.content
        data = _parse_json_object(raw)
        return ImportanceResult(
            score=clamp_importance(data.get("score")),
            reason=str(data.get("reason", "LLM 根据记忆打分规则给出评分。")),
            source="llm",
        )
    except Exception as exc:
        fallback = rule_based_importance(content, role=role, memory_type=memory_type, context=context)
        return ImportanceResult(fallback.score, f"LLM 打分失败，使用规则兜底：{exc}", "rule")


def score_memory_importance(
    content: str,
    role: Optional[str] = None,
    memory_type: str = "working",
    context: Optional[str] = None,
    use_llm: bool = False,
    config: Optional[AppConfig] = None,
    client: Any = None,
) -> ImportanceResult:
    if use_llm:
        return llm_importance(
            content=content,
            role=role,
            memory_type=memory_type,
            context=context,
            config=config,
            client=client,
        )
    return rule_based_importance(
        content=content,
        role=role,
        memory_type=memory_type,
        context=context,
    )


def score_overflow_memory(
    memory: Dict[str, Any],
    context: Optional[str] = None,
    use_llm: bool = True,
    config: Optional[AppConfig] = None,
    client: Any = None,
) -> ImportanceResult:
    """Score working-memory overflow before it is consolidated into episodic memory."""
    return score_memory_importance(
        content=memory.get("content", ""),
        role=memory.get("role"),
        memory_type="episodic",
        context=context,
        use_llm=use_llm,
        config=config,
        client=client,
    )
