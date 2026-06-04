from __future__ import annotations

from typing import List

from sentinel.detectors.rule_detector import INPUT_RULES, detect_rules
from sentinel.schemas import SecurityFinding


class JailbreakDetector:
    def detect(self, text: str) -> List[SecurityFinding]:
        return [
            finding for finding in detect_rules(text, INPUT_RULES)
            if finding.risk_type == "jailbreak"
        ]
