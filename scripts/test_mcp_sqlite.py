"""
测试 sqlite MCP server 是否可用。

准备：
pip install -r requirements.txt

运行：
python scripts/test_mcp_sqlite.py
"""

import asyncio
import sqlite3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import get_config
from mcp_manager import MCPManager


def prepare_demo_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY, title TEXT, content TEXT)")
        conn.execute(
            "INSERT OR IGNORE INTO notes(id, title, content) VALUES (1, 'SENet', 'SE block models channel dependency')"
        )
        conn.commit()
    finally:
        conn.close()


async def main() -> int:
    config = get_config()
    db_path = Path(config.mcp_sqlite_db_path).resolve()
    prepare_demo_db(db_path)

    manager = MCPManager(config=config)
    try:
        probe = await manager.probe_server("sqlite")
        print(f"sqlite 工具数量：{probe['tool_count']}")
        print("工具列表：", ", ".join(probe["tools"]))

        result = await manager.call_tool("sqlite", "list_tables", {})
        if not result.success:
            print(f"调用失败：{result.error or result.content}")
            return 2
        print("list_tables 调用成功，结果预览：")
        print(result.content[:1000])
        return 0
    except Exception as exc:
        print(f"sqlite MCP 测试失败：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
