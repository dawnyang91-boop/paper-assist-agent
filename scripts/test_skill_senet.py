"""
测试本地 skill 是否可创建、导入和使用。

运行：
python scripts/test_skill_senet.py
"""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import AppConfig
from tools.skill_manager import SkillManager


def main() -> int:
    config = AppConfig(
        skills_enabled=True,
        skills_package="skills",
        skills_enabled_names="senet_explainer",
    )
    manager = SkillManager(config=config)
    print("已导入 skills：")
    for skill in manager.list_skills():
        print(f"- {skill['name']}: {skill['description']}")

    memories = manager.retrieve_context("请问 SENet 的核心贡献是什么？")
    print("\nSkill 输出：")
    for memory in memories:
        print(f"{memory['role']}: {memory['content']}")
    return 0 if memories else 1


if __name__ == "__main__":
    raise SystemExit(main())
