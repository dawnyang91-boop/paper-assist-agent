import os
import hashlib
import re
import time
import uuid
from typing import List, Dict, Optional

from config import AppConfig, get_config
from rag.embedding_service import EmbeddingService
from rag.qdrant_utils import ensure_collection_vector_size

def files_in_directory(directory):
    """列出目录下的所有文件, 并以绝对路径列表返回"""
    for root, dirs, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file)


class DocumentConverter:
    def __init__(self, config: Optional[AppConfig] = None):
        """
        初始化转换器。
        如果需要处理图片 (png, jpg)，建议传入大模型的 API Key。
        MarkItDown 默认使用 OpenAI 的接口标准来进行多模态图片解析。
        """
        try:
            from markitdown import MarkItDown
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("未安装 markitdown 或 openai，请先运行 `pip install -r requirements.txt`。") from exc

        self.config = config or get_config()
        self.md_basic = MarkItDown()
        
        # 如果配置了 LLM，则初始化一个带视觉能力的解析器，专供图片使用
        self.md_with_llm = None
        if self.config.openai_api_key:
            client = OpenAI(
                api_key=self.config.openai_api_key,
                base_url=self.config.openai_base_url,
            )
            self.md_with_llm = MarkItDown(llm_client=client, llm_model=self.config.model_name)
        else:
            print("[警告] 未配置 OPENAI_API_KEY，图片文件将无法提取文本内容或描述。")

    def convert_to_markdown(self, file_path: str) -> str:
        """
        将支持的各类文件统一转化为 Markdown 字符串
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件未找到: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        # ==========================================
        # 1. 纯文本与代码类 (.md, .py)
        # ==========================================
        if ext in ['.md', '.py']:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if ext == '.py':
                return f"```python\n{content}\n```"
            return content

        # ==========================================
        # 2. 图像类 (.png, .jpg, .jpeg)
        # ==========================================
        if ext in ['.png', '.jpg', '.jpeg']:
            if self.md_with_llm:
                print(f"正在使用 LLM 视觉模型解析图片: {file_path} ...")
                result = self.md_with_llm.convert(file_path)
                return result.text_content
            else:
                return f"*[提示: 检测到图片 {os.path.basename(file_path)}，但未配置 LLM API，无法提取文本或描述]*"

        # # ==========================================
        # # 3. 矢量图 (.svg)
        # # ==========================================
        # if ext == '.svg':
        #     # SVG 本质是 XML，直接读取对于 RAG 通常意义不大（全是坐标代码）
        #     # 这里的处理策略是：提取其原始代码，或者你可以根据业务需求跳过
        #     with open(file_path, 'r', encoding='utf-8') as f:
        #         content = f.read()
        #     return f"```xml\n\n{content}\n```"

        # ==========================================
        # 4. 办公文档与 PDF (.pdf, .docx, .xlsx)
        # ==========================================
        supported_by_markitdown = ['.pdf', '.docx', '.xlsx', '.pptx', '.html', '.csv']
        if ext in supported_by_markitdown:
            try:
                result = self.md_basic.convert(file_path)
                return result.text_content
            except Exception as e:
                if ext == ".pdf":
                    fallback_text = self._convert_pdf_with_pymupdf(file_path)
                    if fallback_text:
                        return fallback_text
                return f"*[解析 {ext} 文件失败: {str(e)}]*"

        # 兜底处理
        return f"*[不支持的文件格式: {ext}]*"

    def _convert_pdf_with_pymupdf(self, file_path: str) -> str:
        """Fallback PDF text extraction when MarkItDown PDF extras are missing."""
        try:
            import fitz
        except ImportError:
            return ""

        pages = []
        try:
            with fitz.open(file_path) as document:
                for index, page in enumerate(document, start=1):
                    text = page.get_text("text").strip()
                    if text:
                        pages.append(f"## Page {index}\n\n{text}")
        except Exception:
            return ""
        return "\n\n".join(pages).strip()


def is_conversion_failure(markdown_content: str) -> bool:
    text = (markdown_content or "").strip()
    return (
        not text
        or text.startswith("*[解析 ")
        or text.startswith("*[不支持的文件格式:")
        or text.startswith("*[提示: 检测到图片")
    )

def split_paragraphs_with_headings(text: str) -> List[Dict]:
    """根据标题层次分割段落，保持语义完整性"""
    lines = text.splitlines()
    heading_stack: List[str] = []
    paragraphs: List[Dict] = []
    buf: List[str] = []
    char_pos = 0
    
    def flush_buf(end_pos: int):
        if not buf:
            return
        content = "\n".join(buf).strip()
        if not content:
            return
        paragraphs.append({
            "content": content,
            "heading_path": " > ".join(heading_stack) if heading_stack else None,
            "start": max(0, end_pos - len(content)),
            "end": end_pos,
        })
    
    for ln in lines:
        raw = ln
        if raw.strip().startswith("#"):
            # 处理标题行
            flush_buf(char_pos)
            level = len(raw) - len(raw.lstrip('#'))
            title = raw.lstrip('#').strip()
            
            if level <= 0:
                level = 1
            if level <= len(heading_stack):
                heading_stack = heading_stack[:level-1]
            heading_stack.append(title)
            
            char_pos += len(raw) + 1
            continue
        
        # 段落内容累积
        if raw.strip() == "":
            flush_buf(char_pos)
            buf = []
        else:
            buf.append(raw)
        char_pos += len(raw) + 1
    
    flush_buf(char_pos)
    
    if not paragraphs:
        paragraphs = [{"content": text, "heading_path": None, "start": 0, "end": len(text)}]
    
    return paragraphs

def chunk_paragraphs_by_tokens(
    paragraphs: List[Dict],
    max_tokens: int = 2000,
    overlap_tokens: int = 400,
    config: Optional[AppConfig] = None,
) -> List[Dict]:
    """
    基于 Token 数量的智能分块函数。
    将 paragraphs 列表中的内容组合成一个个大块 (chunk)，保证不跨越 max_tokens。
    每一个块的最开始都会加入上一块的尾部部分作为重叠块，建立块与块之前的语义联系。
    主要逻辑如下：
    获取段落附带信息与 Token 数量：函数遍历所有的 paragraphs，计算单段文字加上前置的 heading_path 一共占用多少 token（默认使用 tiktoken，如果没有安装会优雅降级用 len 兜底估算）。
    容量控制：当往当前 chunk 增加一个新段落导致总 Token 数超过 max_tokens 阈值时，我们就把当前 chunk 输出并保存。
    关联重叠块：截断时，算法将从刚好“满”的 chunk 中的末尾开始倒序回溯寻找能够拼凑在 overlap_tokens 限定值内的段落。这些末位段落将作为重叠的内容提前放进下一个 chunk 的最前面，从而串联起切片前后的逻辑连续性。
    长单段保护：即使有些极端单段文字超过了 max_tokens，函数会在新开 chunk 为空的情况下强行放入该长段，保证整个流程不会陷入死循环。
    """
    try:
        import tiktoken
        try:
            cfg = config or get_config()
            model_name = cfg.embed_model_name or cfg.model_name
            encoding = tiktoken.encoding_for_model(model_name)
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")
        def count_tokens(text: str) -> int:
            return len(encoding.encode(text))
    except Exception:
        # 如果未安装 tiktoken，按约 1个中文字符/英文单词 = 1个Token 的粗略方式估算
        print("[警告] tiktoken 不可用或 tokenizer 加载失败，将使用字符串长度粗略估算 Token 数量。")
        def count_tokens(text: str) -> int:
            return len(text)

    # 辅助函数：构造段落带有上下文路径的文本
    def get_para_text(para):
        heading = f"[{para['heading_path']}]\n" if para.get('heading_path') else ""
        return f"{heading}{para['content']}\n\n"

    def make_chunk(chunk_paragraphs: List[Dict], token_count: int) -> Dict:
        combined_text = "".join([get_para_text(p) for p in chunk_paragraphs]).strip()
        heading_paths = []
        for p in chunk_paragraphs:
            heading_path = p.get("heading_path")
            if heading_path and heading_path not in heading_paths:
                heading_paths.append(heading_path)
        starts = [p.get("start") for p in chunk_paragraphs if p.get("start") is not None]
        ends = [p.get("end") for p in chunk_paragraphs if p.get("end") is not None]
        return {
            "content": combined_text,
            "tokens": token_count,
            "heading_paths": heading_paths,
            "start": min(starts) if starts else None,
            "end": max(ends) if ends else None,
        }

    chunks = []
    current_chunk_paragraphs = []
    current_tokens = 0
    has_new_paragraphs = False
    
    i = 0
    while i < len(paragraphs):
        para = paragraphs[i]
        para_text = get_para_text(para)
        para_tokens = count_tokens(para_text)

        # 1. 检查加入当前段落是否会超限。
        # 如果当前块没有加入过新段落 (可能只有重叠块或为空)，即使超限也必须强制加入以避免死循环。
        if current_tokens + para_tokens <= max_tokens or not has_new_paragraphs:
            current_chunk_paragraphs.append(para)
            current_tokens += para_tokens
            has_new_paragraphs = True
            i += 1
        else:
            # 2. 超限，保存当前块
            chunks.append(make_chunk(current_chunk_paragraphs, current_tokens))
            
            # 3. 计算重叠部分：从当前已满块的末尾往前取作为下一块的开头，直到达到 overlap_tokens
            overlap_paragraphs = []
            overlap_current_tokens = 0
            for p in reversed(current_chunk_paragraphs):
                p_text = get_para_text(p)
                p_tokens = count_tokens(p_text)
                if overlap_current_tokens + p_tokens <= overlap_tokens:
                    overlap_paragraphs.insert(0, p)
                    overlap_current_tokens += p_tokens
                else:
                    break
            
            # 4. 开启新块，将上一块的重叠内容先放进去。注意 `i` 不变，下一次循环继续处理由于超限没放进当前块的这个 para
            current_chunk_paragraphs = list(overlap_paragraphs)
            current_tokens = overlap_current_tokens
            has_new_paragraphs = False

    # 收尾处理：若最后还有剩余段落未完成封块
    if current_chunk_paragraphs:
        chunk = make_chunk(current_chunk_paragraphs, current_tokens)
        # 避免全重叠冗余提交
        if not chunks or chunk["content"] not in chunks[-1]["content"]:
            chunks.append(chunk)

    return chunks


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _point_id(source_file: Optional[str], chunk_index: int, content_hash: str) -> str:
    namespace = source_file or "unknown-source"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{namespace}:{chunk_index}:{content_hash}"))


def _content_hash_exists(qdrant, collection_name: str, content_hash: str) -> bool:
    try:
        from qdrant_client.http import models

        points, _ = qdrant.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="content_hash",
                        match=models.MatchValue(value=content_hash),
                    )
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return bool(points)
    except Exception:
        return False


def _extract_chunk_metadata(content: str, source_file: Optional[str]) -> Dict:
    file_name = os.path.basename(source_file) if source_file else None
    title = _extract_title(content, file_name)
    metadata_text = "\n".join(item for item in [file_name or "", title or "", content[:2000]] if item)
    acronyms = _unique_ordered(re.findall(r"\b[A-Z][A-Z0-9_-]{2,}\b", metadata_text))
    model_names = _unique_ordered(
        acronyms
        + re.findall(r"\b[A-Z][a-z]+(?:[A-Z][A-Za-z0-9]+)+(?:[-_][A-Za-z0-9]+)?\b", metadata_text)
        + re.findall(r"\b[A-Za-z]+(?:-[A-Za-z0-9]+)+\b", metadata_text)
    )
    keywords = _unique_ordered([
        *acronyms,
        *model_names,
        *re.findall(r"[A-Za-z][A-Za-z0-9_.+-]{2,}", os.path.splitext(file_name or "")[0]),
    ])[:20]
    return {
        "file_name": file_name,
        "paper_title": title,
        "acronyms": acronyms[:20],
        "model_names": model_names[:20],
        "keywords": keywords,
    }


def _extract_title(content: str, file_name: Optional[str]) -> Optional[str]:
    for line in (content or "").splitlines()[:20]:
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or None
        if 8 <= len(stripped) <= 160 and not stripped.startswith(("-", "|")):
            return stripped
    if file_name:
        return os.path.splitext(file_name)[0].replace("_", " ").replace("-", " ").strip() or None
    return None


def _unique_ordered(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        cleaned = (item or "").strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def _embed_with_retry(client, embed_kwargs: Dict, max_try: int, wait_seconds: float, backoff: float):
    max_try = max(1, int(max_try or 1))
    wait_seconds = max(0.0, float(wait_seconds or 0.0))
    backoff = max(1.0, float(backoff or 1.0))

    last_error = None
    for attempt in range(1, max_try + 1):
        try:
            return client.embeddings.create(**embed_kwargs)
        except Exception as exc:
            last_error = exc
            if attempt >= max_try:
                break
            delay = wait_seconds * (backoff ** (attempt - 1))
            print(f"    [重试] 嵌入请求失败，第 {attempt}/{max_try} 次：{exc}")
            if delay > 0:
                print(f"    [重试] {delay:.1f}s 后再次尝试...")
                time.sleep(delay)

    raise last_error


def build_chunk_payload(chunk: Dict, chunk_index: int, source_file: Optional[str] = None) -> Dict:
    content = chunk["content"]
    file_type = os.path.splitext(source_file)[1].lower() if source_file else None
    content_hash = _content_hash(content)
    payload = {
        "page_content": content,
        "content": content,
        "tokens": chunk.get("tokens"),
        "chunk_index": chunk_index,
        "source_file": source_file,
        "file_type": file_type,
        "heading_paths": chunk.get("heading_paths", []),
        "start": chunk.get("start"),
        "end": chunk.get("end"),
        "content_hash": content_hash,
        "created_at": time.time(),
    }
    payload.update(_extract_chunk_metadata(content, source_file))
    return payload


def store_chunks_to_qdrant(
    chunks: List[Dict],
    collection_name: Optional[str] = None,
    source_file: Optional[str] = None,
    config: Optional[AppConfig] = None,
    qdrant_client=None,
):
    """
    将分块后的文本嵌入并存储到 Qdrant 中。
    需要在 .env 中配置 EMBED_API_KEY 或 OPENAI_API_KEY。
    """
    cfg = config or get_config()
    collection_name = collection_name or cfg.rag_collection_name

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import PointStruct
    except ImportError:
        print("[错误] 未安装 qdrant_client，请使用 `pip install qdrant-client` 安装。")
        return

    embedding_service = EmbeddingService(config=cfg)
    embedding_model = cfg.embed_model_name
    
    print(f"\n[*] 准备连接 Qdrant collection: {collection_name} ...")
    qdrant = qdrant_client or QdrantClient(
        url=cfg.qdrant_url,
        api_key=cfg.qdrant_api_key,
        timeout=cfg.qdrant_timeout,
    )
    
    if not chunks:
        print("[*] 没有可处理的块。")
        return
        
    print(f"[*] 开始对 {len(chunks)} 个块进行文本嵌入 (使用模型: {embedding_model}) ...")
    print(f"[*] 嵌入失败重试策略: max_try={cfg.embed_max_try}, wait={cfg.embed_retry_wait_seconds}s, backoff={cfg.embed_retry_backoff}")
    
    points = []
    collection_checked = False
    seen_content_hashes = set()
    
    for idx, chunk in enumerate(chunks):
        content = chunk["content"]
        try:
            payload = build_chunk_payload(chunk, idx, source_file=source_file)
            if payload["content_hash"] in seen_content_hashes:
                print(f"  - 跳过重复块 {idx+1}/{len(chunks)} (本次任务内 content_hash 重复)")
                continue
            seen_content_hashes.add(payload["content_hash"])
            if _content_hash_exists(qdrant, collection_name, payload["content_hash"]):
                print(f"  - 跳过重复块 {idx+1}/{len(chunks)} (content_hash 已存在)")
                continue

            vector = _embed_text_with_retry(
                embedding_service=embedding_service,
                text=content,
                max_try=cfg.embed_max_try,
                wait_seconds=cfg.embed_retry_wait_seconds,
                backoff=cfg.embed_retry_backoff,
            )
            
            if not collection_checked:
                ensure_collection_vector_size(qdrant, collection_name, len(vector))
                collection_checked = True
            
            points.append(PointStruct(
                id=_point_id(source_file, idx, payload["content_hash"]),
                vector=vector,
                payload=payload,
            ))
            print(f"  - 成功嵌入块 {idx+1}/{len(chunks)} (Tokens: {chunk['tokens']})")
            
        except Exception as e:
            print(f"  [错误] 嵌入第 {idx+1} 个块时发生错误: {str(e)}")
    
    if points:
        print(f"[*] 开始将 {len(points)} 个向量存入 Qdrant 集合 '{collection_name}' ...")
        qdrant.upsert(
            collection_name=collection_name,
            points=points
        )
        print(f"[*] Qdrant 数据存储完成！")
    else:
        print("[*] 没有成功生成的向量可供存储。")


def _embed_text_with_retry(
    embedding_service: EmbeddingService,
    text: str,
    max_try: int,
    wait_seconds: float,
    backoff: float,
) -> List[float]:
    max_try = max(1, int(max_try or 1))
    wait_seconds = max(0.0, float(wait_seconds or 0.0))
    backoff = max(1.0, float(backoff or 1.0))
    last_error = None
    for attempt in range(1, max_try + 1):
        try:
            return embedding_service.embed_text(text)
        except Exception as exc:
            last_error = exc
            if attempt >= max_try:
                break
            delay = wait_seconds * (backoff ** (attempt - 1))
            print(f"    [重试] 嵌入请求失败，第 {attempt}/{max_try} 次：{exc}")
            if delay > 0:
                print(f"    [重试] {delay:.1f}s 后再次尝试...")
                time.sleep(delay)
    raise last_error

# ==========================================
# 测试用例
# ==========================================
if __name__ == "__main__":
    converter = DocumentConverter()
    file_paths = list(files_in_directory("./test_files"))
    for test_file in file_paths:
        print(f"\n--- 处理文件: {test_file} ---")
        markdown_content = converter.convert_to_markdown(test_file)
        
        # 1. 切分为带有层级的极小段落
        paragraphs = split_paragraphs_with_headings(markdown_content)
        
        # 2. 聚合成符合一定 token 上限（如2000）的大块，且包含重叠部分
        chunks = chunk_paragraphs_by_tokens(paragraphs, max_tokens=2000, overlap_tokens=400)
        
        print(f"\n[*] 共划分为 {len(chunks)} 个 Chunk：")
        for i, chunk in enumerate(chunks):
            print(f"  - Chunk {i+1}: 预估Tokens={chunk['tokens']}, 预览={chunk['content'][:50]}...")
            
        # 3. 存储进 Qdrant 的指定集合中
        # 你可以为不同的文件建立不同名称的 collection，或统一放一起（加上文件名 Payload）
        store_chunks_to_qdrant(chunks, source_file=test_file)
