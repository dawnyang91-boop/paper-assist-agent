from __future__ import annotations

from sentinel.config import SentinelConfig
from sentinel.detectors.jailbreak_detector import JailbreakDetector
from sentinel.detectors.prompt_injection_detector import PromptInjectionDetector
from sentinel.schemas import SecurityDecision, decision_from_findings


class InputGuard:
    def __init__(self, config: SentinelConfig):
        self.config = config
        self.prompt_injection_detector = PromptInjectionDetector()
        self.jailbreak_detector = JailbreakDetector()

    def check(self, text: str) -> SecurityDecision:
        if not self.config.enabled or not self.config.input_guard_enabled:
            return SecurityDecision(allowed=True, action="allow", risk_score=0.0, sanitized_text=text)
        findings = [
            *self.prompt_injection_detector.detect(text),
            *self.jailbreak_detector.detect(text),
        ]
        return decision_from_findings(
            findings,
            sanitized_text=None,
            warn_threshold=self.config.warn_threshold,
            block_threshold=self.config.block_threshold,
        )
