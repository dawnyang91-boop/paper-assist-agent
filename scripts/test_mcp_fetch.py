"""
测试 fetch MCP server 是否可用。

准备：
pip install -r requirements.txt

运行：
python scripts/test_mcp_fetch.py
python scripts/test_mcp_fetch.py https://example.com
"""

import asyncio
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import get_config
from mcp_manager import MCPManager


async def main() -> int:
    manager = MCPManager(config=get_config())
    try:
        probe = await manager.probe_server("fetch")
        print(f"fetch 工具数量：{probe['tool_count']}")
        print("工具列表：", ", ".join(probe["tools"]))

        url = sys.argv[1] if len(sys.argv) > 1 else "https://www.example.com"
        result = await manager.call_tool("fetch", "fetch", {"url": url})
        if not result.success:
            print(f"调用失败：{result.error or result.content}")
            return 2
        print("fetch 调用成功，结果预览：")
        print(result.content[:1000])
        return 0
    except Exception as exc:
        print(f"fetch MCP 测试失败：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
