import re
from typing import Dict, Optional

from skill_manager import SkillResult


class PaperDeepSummarySkill:
    name = "paper_deep_summary"
    description = "提供论文详细总结、Zotero 笔记和第一性原理论文分析的结构化模板。"
    trigger_keywords = [
        "paper summary",
        "summarize paper",
        "paper summarization",
        "summarize this paper",
        "summarize the paper",
        "article summary",
        "academic summary",
        "detailed paper note",
        "literature note",
        "literature review",
        "zotero",
        "research core",
        "novelty",
        "motivation",
        "limitation",
        "research gap",
        "experiment",
        "论文总结",
        "详细总结",
        "总结论文",
        "总结这篇论文",
        "总结文章",
        "文献总结",
        "文献笔记",
        "论文笔记",
        "论文分析",
        "分析论文",
        "创新点",
        "局限性",
        "研究空白",
        "实验设计",
        "研究核心",
        "个人总结",
        "第一性原理",
    ]

    def should_activate(self, question: str) -> bool:
        lowered = (question or "").lower()
        if any(keyword.lower() in lowered for keyword in self.trigger_keywords):
            return True
        return bool(
            re.search(r"(总结|概括|分析|精读|解读).{0,40}(论文|文章|文献|paper|article)", lowered)
            or re.search(r"(论文|文章|文献|paper|article).{0,40}(总结|概括|分析|精读|解读)", lowered)
        )

    def run(self, question: str, context: Optional[Dict] = None) -> SkillResult:
        content = (
            "Paper deep-summary skill activated.\n"
            "Use this skill when the user asks for a detailed paper summary, Zotero-style paper note, "
            "research novelty analysis, motivation reconstruction, limitations, or research-gap analysis.\n\n"
            "Important safety rule: never hard-code API keys in generated Zotero templates or code. "
            "If code is needed, read credentials from environment variables or the project configuration.\n\n"
            "Final output should be written in Chinese unless the user explicitly asks otherwise.\n\n"
            "Recommended detailed paper-note structure:\n"
            "1. Metadata block\n"
            "- Title, Chinese title if available, publication date, authors, journal/conference, DOI/URL, local attachment link if available.\n"
            "- Abstract or translated abstract.\n\n"
            "2. 研究核心\n"
            "- Task: 这篇文章解决的是什么问题？\n"
            "- 详细阐述论文的核心研究内容、背景问题和具体任务，不要只用一两句话概括。\n"
            "- 说明传统方法遇到的挑战。\n"
            "- Insight & Novelty: 分析作者的 insight 来源、具体 insight、novelty 类型。\n"
            "- 对每个创新点按这个格式描述：【创新点解决的问题是什么】->【受哪个 insight 启发】->【设计了什么创新点】。\n\n"
            "3. Potential Limitations & Research Gaps\n"
            "- 结合核心方法、理论假设与实验设计，分析具体场景边界。\n"
            "- 避免只写“多模态拓展”“数据噪声”等泛泛限制。\n"
            "- 从模型机制、优化目标、理论推导、计算开销、泛化能力、落地可行性分析专属困难。\n"
            "- 筛选可以形成独立研究工作的非通用、具备学术创新性的问题。\n\n"
            "4. 研究内容\n"
            "- 数据：数据集名称、样本规模、预处理方法、特征维度。\n"
            "- 方法：核心算法架构、统计模型、网络结构、分析流程、关键技术细节和推导思路。\n"
            "- 实验：实验设计、baselines、metrics、消融实验、主要定量结果。\n"
            "- 结论：定量/定性结果、科学结论、未来展望。\n\n"
            "5. Motivation\n"
            "- 用第一性原理重构作者想到 general idea 的路径。\n"
            "- 最好用问句表达，例如：既然旧方法存在 X 限制，那能不能从 Y 角度重新建模？\n"
            "- 从问题本质出发，推导最自然、最容易想到本文 idea 的方式。\n\n"
            "6. 个人总结\n"
            "- 重点记录：用户最值得记住的结论、方法和细节。\n"
            "- 待解决：读完后仍不清楚或可继续查证的问题。\n"
            "- 思考启发：可迁移到用户项目或后续研究的启发。\n\n"
            "Output style:\n"
            "- Prefer clear HTML or Markdown sections according to user request.\n"
            "- If HTML is requested, use headings, paragraphs, blockquotes, and lists; do not wrap the result in Markdown code fences unless asked.\n"
            "- Filter obvious OCR noise, headers, footers, references, and corrupted text.\n"
            "- When evidence is missing, say which part is inferred rather than pretending the paper states it."
        )
        return SkillResult(
            name=self.name,
            content=content,
            metadata={
                "question": question,
                "source": "local_skill",
                "template": "paper_deep_summary",
                "sanitized": True,
            },
        )


SKILL = PaperDeepSummarySkill()
