"""Markdown paper-summary tool.

The tool preserves the ideas of the user's Zotero/HTML template but produces a
clean Markdown note that can be saved, versioned, and ingested into RAG.

It can either:
- generate an empty structured template for manual writing, or
- call an OpenAI-compatible LLM using environment variables / explicit safe args.

Never hard-code API keys in this file or generated notes.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .common import clean_text, ensure_dir, fail, ok, safe_filename

DEFAULT_SUMMARY_SYSTEM_PROMPT = """你是一个严谨的学术助手，也是一名第一性原理思考者。
你擅长从问题本质、基本约束、常识和研究动机出发，重构论文的核心 idea、方法设计、实验逻辑和潜在研究空白。
请用中文输出结构化 Markdown。不要输出 HTML，不要包裹 Markdown 代码块。"""

SUMMARY_PROMPT_TEMPLATE = """请仔细阅读以下论文{source_type}，生成一份【详细、深入】的 Markdown 论文总结。

要求：
1. 必须深入总结，不能只用一两句话概括。
2. 多项内容请使用 Markdown 列表，排版清晰。
3. 自行过滤乱码、页眉页脚、参考文献噪声。
4. 当原文没有提供充分信息时，请明确写“原文信息不足”，不要编造。
5. 保留第一性原理分析：从问题本质出发，解释为什么作者会想到这个 general idea。

请严格按照以下 Markdown 结构输出：

## 📜 研究核心

> Task：这篇文章解决的是什么问题？

### ⚙️ 内容

- 详细阐述论文的核心研究内容、试图解决的具体背景问题。不少于 100 字。
- 传统方法在解决这个问题时遇到了什么挑战？

### 💡 Insight & Novelty

- 作者的 Insight 是被什么 Inspiration 启发的？
- 作者的 Insight 究竟是什么？是在什么方面上的 Insight？对于每个 Insight，是哪些 Inspiration 启发的？
- 作者本篇文章的 Novelty 体现在何处？是架构上、方法上、策略上还是实验范式上的创新？
- 对于每一个 Novelty，请严格按这个格式描述：
  - 【创新点解决的问题是什么】→【受哪个 insight 启发】→【设计了什么创新点，尽可能具体描述】

### 🧩 Potential Limitations & Research Gaps

1. 结合论文核心方法、理论假设与实验设计，分析其具体场景边界：该方法仅适用于哪些限定条件？是否存在任务类型、数据分布、模型结构、评价指标上的固有局限？禁止只写“可扩展到多模态”。
2. 抛开“数据噪声、脏数据”这类通用问题，从模型机制、优化目标、理论推导、计算开销、泛化能力、落地可行性角度，分析该方案在当前场景下会遇到哪些专属困难。
3. 从上述局限性中，筛选出非通用、具备学术创新性的问题：哪一点可以通过改进模型架构、修正理论假设、优化实验范式形成新的研究工作，值得独立写成学术论文？

## 🔁 研究内容

### 💧 数据

详细说明研究所用的数据集名称、样本量规模、数据预处理方法或特征维度。如果原文没有说明，请明确指出。

### 👩🏻‍💻 方法

详细描述核心算法架构、数理统计模型、网络结构或分析流程，务必保留关键技术细节和推导思路。

### 🔬 实验

详细概括实验设计、对比基线算法/模型、核心评估指标及消融实验结果。

### 📜 结论

全面总结实验的定量与定性结果，以及该研究最终得出的科学结论和未来展望。

## 🚀 Motivation

请总结这篇文章想到 general idea 的方式，最好以问句形式给出，例如：“既然之前的方法存在 X 限制，那可不可以尝试 Y？”
请遵循第一性原理，从问题本质出发，找到最合理、最容易想到本文 idea 的路径。

论文元数据：
{metadata}

论文源文本：
{content}
"""


def paper_summary_tool(
    paper_metadata: Optional[Dict[str, Any]] = None,
    content: Optional[str] = None,
    content_source: str = "auto",
    use_llm: bool = False,
    model_name: Optional[str] = None,
    api_base_url: Optional[str] = None,
    api_key_env: str = "OPENAI_API_KEY",
    max_input_chars: int = 30000,
    save: bool = True,
    save_dir: str = "data/papers/summaries",
    filename: Optional[str] = None,
    include_personal_section: bool = True,
    **_: Any,
) -> Dict[str, Any]:
    """Generate and optionally save a detailed Markdown paper summary."""
    metadata = paper_metadata or {}
    title = clean_text(metadata.get("title")) or "Untitled Paper"
    source_text = clean_text(content) or clean_text(metadata.get("abstract")) or clean_text(metadata.get("abstractNote"))
    is_full_text = bool(content and len(content) > 500)

    if not source_text:
        return fail(
            "Cannot generate paper summary because no full text or abstract was provided.",
            error="missing content",
            data={
                "status": "missing_content",
                "paper_metadata": metadata,
                "manual_upload_required": True,
                "suggested_action": "Provide parsed paper text or at least an abstract before generating summary.",
            },
        )

    source_text = source_text[: max(1000, int(max_input_chars or 30000))]
    if use_llm:
        generated_body = _generate_summary_with_llm(
            metadata=metadata,
            content=source_text,
            is_full_text=is_full_text,
            model_name=model_name,
            api_base_url=api_base_url,
            api_key_env=api_key_env,
        )
        if not generated_body.get("success"):
            return generated_body
        body_markdown = generated_body["data"]["summary_markdown"]
    else:
        body_markdown = _empty_summary_body(source_text_available=True, is_full_text=is_full_text)

    markdown = _build_summary_markdown(
        metadata=metadata,
        body_markdown=body_markdown,
        include_personal_section=include_personal_section,
    )

    saved_path = None
    if save:
        directory = ensure_dir(save_dir)
        file_name = filename or f"{safe_filename(title)}_summary.md"
        if not file_name.endswith(".md"):
            file_name += ".md"
        path = directory / file_name
        path.write_text(markdown, encoding="utf-8")
        saved_path = str(path)

    return ok(
        {
            "status": "saved" if saved_path else "generated",
            "paper_metadata": metadata,
            "summary_markdown": markdown,
            "saved_path": saved_path,
            "is_full_text": is_full_text,
            "content_source": content_source,
            "used_llm": use_llm,
        },
        message="Paper summary generated.",
    )


def _build_summary_markdown(
    metadata: Dict[str, Any],
    body_markdown: str,
    include_personal_section: bool,
) -> str:
    title = clean_text(metadata.get("title")) or "Untitled Paper"
    title_translation = clean_text(metadata.get("titleTranslation") or metadata.get("title_translation"))
    date = _date_only(metadata.get("date") or metadata.get("publication_date"))
    authors = _format_authors(metadata.get("authors") or metadata.get("creators"))
    venue = clean_text(metadata.get("venue") or metadata.get("publicationTitle") or metadata.get("publication_title"))
    doi = clean_text(metadata.get("doi") or metadata.get("DOI"))
    url = clean_text(metadata.get("url") or metadata.get("landing_page_url"))
    local_link = clean_text(metadata.get("local_path") or metadata.get("attachment") or metadata.get("pdf_url"))
    abstract = clean_text(metadata.get("abstractTranslation") or metadata.get("abstract_translation") or metadata.get("abstract") or metadata.get("abstractNote"))
    note_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    display_title = f"({date}) {title}" if date else title
    if title_translation:
        display_title += f"（{title_translation}）"

    lines: List[str] = [
        "# 📝 论文总结-详细版",
        "",
        f"## {display_title}",
        "",
        "## 📌 论文信息",
        "",
        "| 字段 | 内容 |",
        "|---|---|",
        f"| 作者 | {authors} |",
        f"| 期刊/会议 | {venue or ''} |",
        f"| 发表日期 | {date or ''} |",
        f"| 本地链接 | {local_link or ''} |",
        f"| DOI/URL | {_doi_or_url(doi, url)} |",
        f"| 笔记日期 | {note_date} |",
        "",
    ]
    if abstract:
        lines.extend(["## 摘要", "", f"> {abstract}", ""])

    lines.append(body_markdown.strip())

    if include_personal_section:
        lines.extend([
            "",
            "## 🤔 个人总结",
            "",
            "> Tips：你对哪些内容产生了疑问，你认为可以如何改进？",
            "",
            "### 🙋‍♀️ 重点记录",
            "",
            "- ",
            "",
            "### 📌 待解决",
            "",
            "- ",
            "",
            "### 💭 思考启发",
            "",
            "- ",
        ])
    return "\n".join(lines).strip() + "\n"


def _empty_summary_body(source_text_available: bool, is_full_text: bool) -> str:
    source_note = "已提供全文，可基于全文填写。" if is_full_text else "仅提供摘要或短文本，部分细节需要后续补全文。"
    return f"""## 📜 研究核心

> Task：这篇文章解决的是什么问题？

### ⚙️ 内容

- 待填写：详细阐述论文的核心研究内容、试图解决的具体背景问题。不少于 100 字。
- 待填写：传统方法在解决这个问题时遇到了什么挑战？
- 内容来源状态：{source_note}

### 💡 Insight & Novelty

- 待填写：作者的 Insight 是被什么 Inspiration 启发的？
- 待填写：作者的 Insight 究竟是什么？是在什么方面上的 Insight？
- 待填写：作者本篇文章的 Novelty 体现在何处？
- 待填写：
  - 【创新点解决的问题是什么】→【受哪个 insight 启发】→【设计了什么创新点，尽可能具体描述】

### 🧩 Potential Limitations & Research Gaps

1. 待填写：结合论文核心方法、理论假设与实验设计，分析其具体场景边界。
2. 待填写：从模型机制、优化目标、理论推导、计算开销、泛化能力、落地可行性角度分析专属困难。
3. 待填写：筛选出非通用、具备学术创新性、可独立发展成论文的问题。

## 🔁 研究内容

### 💧 数据

- 待填写：数据集名称、样本量规模、预处理方法、特征维度。

### 👩🏻‍💻 方法

- 待填写：核心算法架构、数理统计模型、网络结构或分析流程。

### 🔬 实验

- 待填写：实验设计、Baselines、Metrics、消融实验和主要结果。

### 📜 结论

- 待填写：定量/定性结果、科学结论、未来展望。

## 🚀 Motivation

- 待填写：既然之前的方法存在什么限制，那可不可以从什么角度重新建模或改进？
"""


def _generate_summary_with_llm(
    metadata: Dict[str, Any],
    content: str,
    is_full_text: bool,
    model_name: Optional[str],
    api_base_url: Optional[str],
    api_key_env: str,
) -> Dict[str, Any]:
    api_key = os.getenv(api_key_env or "OPENAI_API_KEY")
    if not api_key:
        return fail(
            "LLM summary generation requested but API key is missing.",
            error=f"missing environment variable: {api_key_env}",
            data={"status": "missing_api_key", "api_key_env": api_key_env},
        )
    try:
        from openai import OpenAI
    except ImportError as exc:
        return fail("OpenAI SDK is not installed.", error=str(exc), data={"status": "missing_dependency"})

    client = OpenAI(api_key=api_key, base_url=api_base_url or os.getenv("OPENAI_BASE_URL"))
    model = model_name or os.getenv("MODEL_NAME") or "gpt-4o-mini"
    prompt = SUMMARY_PROMPT_TEMPLATE.format(
        source_type="全文提取的文本" if is_full_text else "摘要或短文本",
        metadata=metadata,
        content=content,
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": DEFAULT_SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=3000,
        )
        summary = response.choices[0].message.content or ""
        return ok({"summary_markdown": summary.strip(), "model": model}, message="LLM summary generated.")
    except Exception as exc:
        return fail("LLM summary generation failed.", error=str(exc), data={"status": "llm_failed", "model": model})


def _format_authors(authors: Any) -> str:
    if not authors:
        return ""
    if isinstance(authors, str):
        return authors
    formatted = []
    for author in list(authors)[:10]:
        if isinstance(author, str):
            formatted.append(author)
        elif isinstance(author, dict):
            name = clean_text(author.get("name"))
            if not name:
                name = clean_text(" ".join(part for part in [author.get("firstName"), author.get("lastName")] if part))
            if not name:
                name = clean_text(" ".join(part for part in [author.get("given"), author.get("family")] if part))
            if name:
                formatted.append(name)
    if isinstance(authors, list) and len(authors) > 10:
        formatted.append("et al.")
    return "; ".join(formatted)


def _date_only(value: Any) -> str:
    text = clean_text(value) or ""
    return text.split("T", 1)[0]


def _doi_or_url(doi: Optional[str], url: Optional[str]) -> str:
    if doi:
        return f"https://doi.org/{doi}"
    return url or ""
