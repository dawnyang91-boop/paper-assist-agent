from __future__ import annotations

import json
from pathlib import Path
from typing import List

from sentinel.evaluators.report_writer import ReportWriter
from sentinel.evaluators.scorer import SentinelScorer
from sentinel.evaluators.target_adapters import TargetAdapter
from sentinel.schemas import EvalCase, EvalResult


class SentinelRunner:
    def __init__(self, adapter: TargetAdapter, scorer: SentinelScorer | None = None):
        self.adapter = adapter
        self.scorer = scorer or SentinelScorer()

    def run_cases(self, cases: List[EvalCase]) -> List[EvalResult]:
        results: List[EvalResult] = []
        for case in cases:
            try:
                response = self.adapter.ask(case)
                results.append(self.scorer.score(case, response))
            except Exception as exc:
                results.append(EvalResult(
                    case_id=case.case_id,
                    passed=False,
                    score=0.0,
                    target_answer="",
                    findings=[],
                    metrics={"error": str(exc), "expected_behavior": case.expected_behavior},
                ))
        return results

    def run_dataset(self, dataset_path: str | Path) -> List[EvalResult]:
        return self.run_cases(load_cases(dataset_path))

    def run_and_report(self, dataset_path: str | Path, output_base: str | Path) -> dict:
        results = self.run_dataset(dataset_path)
        return ReportWriter().write(results, output_base)


def load_cases(dataset_path: str | Path) -> List[EvalCase]:
    path = Path(dataset_path)
    text = path.read_text(encoding="utf-8")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Dataset is not JSON and PyYAML is not installed.") from exc
        raw = yaml.safe_load(text)
    if isinstance(raw, dict) and isinstance(raw.get("includes"), list):
        cases: List[EvalCase] = []
        for include in raw["includes"]:
            include_path = Path(str(include))
            if not include_path.is_absolute():
                include_path = path.parent / include_path
            cases.extend(load_cases(include_path))
        return cases
    return [EvalCase.from_dict(item) for item in raw or []]
