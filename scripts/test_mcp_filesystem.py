"""
测试 filesystem MCP server 是否可用。

准备：
npm/npx 可用，且网络可拉取 @modelcontextprotocol/server-filesystem。

运行：
MCP_ENABLED=true python scripts/test_mcp_filesystem.py
"""

import asyncio
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import get_config
from tools.mcp_manager import MCPManager


async def main() -> int:
    manager = MCPManager(config=get_config())
    try:
        probe = await manager.probe_server("filesystem")
        print(f"filesystem 工具数量：{probe['tool_count']}")
        print("工具列表：", ", ".join(probe["tools"]))

        result = await manager.call_tool(
            "filesystem",
            "list_directory",
            {"path": str(Path(get_config().mcp_filesystem_root).resolve())},
        )
        if not result.success:
            print(f"调用失败：{result.error or result.content}")
            return 2
        print("list_directory 调用成功，结果预览：")
        print(result.content[:1000])
        return 0
    except Exception as exc:
        print(f"filesystem MCP 测试失败：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
