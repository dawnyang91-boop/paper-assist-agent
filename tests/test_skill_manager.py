from config import AppConfig
from skill_manager import SkillManager


def test_skill_manager_loads_configured_skill():
    config = AppConfig(
        skills_enabled=True,
        skills_package="skills",
        skills_enabled_names="senet_explainer",
    )

    manager = SkillManager(config=config)

    skills = manager.list_skills()
    assert skills[0]["name"] == "senet_explainer"


def test_skill_manager_returns_context_for_triggered_skill():
    config = AppConfig(
        skills_enabled=True,
        skills_package="skills",
        skills_enabled_names="senet_explainer",
    )
    manager = SkillManager(config=config)

    memories = manager.retrieve_context("SENet 的 SE block 有什么作用？")

    assert len(memories) == 1
    assert memories[0]["type"] == "skill"
    assert "Squeeze-and-Excitation" in memories[0]["content"]


def test_skill_manager_activates_explicitly_named_paper_summary_skill():
    config = AppConfig(
        skills_enabled=True,
        skills_package="skills",
        skills_enabled_names="paper_deep_summary",
    )
    manager = SkillManager(config=config)

    memories = manager.retrieve_context("请使用 PaperDeepSummarySkill 总结OneTrans这篇文章")

    assert len(memories) == 1
    assert memories[0]["role"] == "skill:paper_deep_summary"
    assert "Paper deep-summary skill activated" in memories[0]["content"]


def test_skill_manager_activates_paper_summary_with_title_between_words():
    config = AppConfig(
        skills_enabled=True,
        skills_package="skills",
        skills_enabled_names="paper_deep_summary",
    )
    manager = SkillManager(config=config)

    memories = manager.retrieve_context("请总结 OneTrans 这篇文章的创新点")

    assert len(memories) == 1
    assert memories[0]["role"] == "skill:paper_deep_summary"
