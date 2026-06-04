POLLUTED_CONTEXT_MARKERS = (
    "未配置 OPENAI_API_KEY",
    "以下是基于检索上下文的摘要式回答",
    "当前没有检索到可用文档片段",
    "无法基于私域知识库给出可靠答案",
    "根据您提供的上下文，我无法回答",
    "根据你提供的上下文，我无法回答",
    "本地文档/记忆中没有找到足够依据；同时当前 LLM 客户端不可用",
    "需要提供相关",
)


def is_polluted_context(content: str) -> bool:
    text = (content or "").strip()
    if not text:
        return False
    return any(marker in text for marker in POLLUTED_CONTEXT_MARKERS)
