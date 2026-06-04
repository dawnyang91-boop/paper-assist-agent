from typing import Dict, Optional

from skill_manager import SkillResult


class SENetExplainerSkill:
    name = "senet_explainer"
    description = "为 SENet / SE block 相关问题补充固定解释框架。"
    trigger_keywords = ["SENet", "SE block", "Squeeze-and-Excitation", "通道注意力", "特征重标定"]

    def should_activate(self, question: str) -> bool:
        lowered = question.lower()
        return any(keyword.lower() in lowered for keyword in self.trigger_keywords)

    def run(self, question: str, context: Optional[Dict] = None) -> SkillResult:
        content = (
            "SENet 相关回答建议框架："
            "1. 先说明核心贡献是通过 Squeeze-and-Excitation block 显式建模通道依赖；"
            "2. 再说明 squeeze 使用全局信息压缩空间维度，excitation 学习通道权重；"
            "3. 最后说明特征重标定会增强有用通道、抑制不重要通道，可作为轻量模块嵌入 CNN。"
        )
        return SkillResult(
            name=self.name,
            content=content,
            metadata={"question": question, "source": "local_skill"},
        )


SKILL = SENetExplainerSkill()
