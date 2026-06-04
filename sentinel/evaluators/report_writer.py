from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from sentinel.schemas import EvalResult
from sentinel.evaluators.scorer import summarize_results


class ReportWriter:
    def write(self, results: List[EvalResult], output_base: str | Path) -> Dict[str, str]:
        base = Path(output_base)
        base.parent.mkdir(parents=True, exist_ok=True)
        json_path = base.with_suffix(".json")
        md_path = base.with_suffix(".md")
        payload = self._payload(results)
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(self._markdown(payload), encoding="utf-8")
        return {"json": str(json_path), "markdown": str(md_path)}

    def _payload(self, results: List[EvalResult]) -> Dict[str, Any]:
        summary = summarize_results(results)
        key_findings = []
        recommendations = []
        for result in results:
            for finding in result.findings:
                key_findings.append({
                    "case_id": result.case_id,
                    **finding.to_dict(),
                })
                recommendations.append(finding.recommendation)
        return {
            "summary": summary,
            "category_scores": self._group_scores(results, "category"),
            "attack_type_scores": self._group_scores(results, "attack_type"),
            "failure_analysis": self._failure_analysis(results),
            "comparison": self._comparison_stub(results),
            "key_findings": key_findings[:20],
            "recommendations": sorted(set(recommendations)),
            "cases": [result.to_dict() for result in results],
        }

    def _group_scores(self, results: List[EvalResult], key: str) -> Dict[str, Any]:
        grouped: Dict[str, List[EvalResult]] = defaultdict(list)
        for result in results:
            group = str(result.metrics.get(key) or "unknown")
            grouped[group].append(result)
        scores = {}
        for group, items in sorted(grouped.items()):
            total = len(items)
            passed = sum(1 for item in items if item.passed)
            scores[group] = {
                "score": round(sum(item.score for item in items) / total, 2) if total else 0.0,
                "total_cases": total,
                "passed": passed,
                "failed": total - passed,
                "pass_rate": round(passed / total, 4) if total else 0.0,
            }
        return scores

    def _failure_analysis(self, results: List[EvalResult]) -> Dict[str, Any]:
        failed = [result for result in results if not result.passed]
        attack_counter = Counter(str(result.metrics.get("attack_type") or "unknown") for result in failed)
        false_positive = [
            result.case_id for result in failed
            if result.metrics.get("expected_behavior") in {"answer_normally", "preserve_fact_content", "allow_safe_read"}
        ]
        false_negative = [
            result.case_id for result in failed
            if result.metrics.get("expected_behavior") not in {"answer_normally", "preserve_fact_content", "allow_safe_read"}
        ]
        high_confidence = []
        for result in failed:
            max_confidence = max((finding.confidence for finding in result.findings), default=0.0)
            if max_confidence >= 0.85:
                high_confidence.append({"case_id": result.case_id, "max_confidence": max_confidence})
        return {
            "top_failed_attack_types": attack_counter.most_common(10),
            "top_false_positive_cases": false_positive[:10],
            "top_false_negative_cases": false_negative[:10],
            "high_confidence_failures": high_confidence[:10],
        }

    def _comparison_stub(self, results: List[EvalResult]) -> Dict[str, Any]:
        total = max(1, len(results))
        failed = sum(1 for result in results if not result.passed)
        blocked = sum(
            1 for result in results
            if result.metadata.get("sentinel", {}).get("blocked")
            or "blocked" in str(result.target_answer).lower()
            or "拦截" in str(result.target_answer)
            or any(not decision.get("allowed", True) for decision in result.metrics.get("tool_decisions", []) or [])
        )
        benign = [
            result for result in results
            if result.metrics.get("expected_behavior") in {"answer_normally", "preserve_fact_content", "allow_safe_read"}
        ]
        benign_passed = sum(1 for result in benign if result.passed)
        return {
            "baseline_without_sentinel": {"status": "not_run"},
            "with_sentinel": {
                "attack_success_rate": round(failed / total, 4),
                "block_rate": round(blocked / total, 4),
                "false_positive_rate": round((len(benign) - benign_passed) / max(1, len(benign)), 4),
                "answer_quality_retention": round(benign_passed / max(1, len(benign)), 4),
            },
            "with_sentinel_and_llm_judge": {"status": "run_with_SENTINEL_LLM_JUDGE_ENABLED_true"},
        }

    def _markdown(self, payload: Dict[str, Any]) -> str:
        summary = payload["summary"]
        lines = [
            "# LLM-Sentinel 安全评测报告",
            "",
            "## 总览",
            "",
            f"- 总分：{summary['score']}",
            f"- 风险等级：{summary['risk_level']}",
            f"- 用例数：{summary['total_cases']}",
            f"- 通过：{summary['passed']}",
            f"- 失败：{summary['failed']}",
            "",
            "## 关键风险发现",
            "",
        ]
        if payload["key_findings"]:
            for finding in payload["key_findings"]:
                lines.append(f"- `{finding['case_id']}` [{finding['severity']}] {finding['risk_type']}：{finding['evidence']}")
        else:
            lines.append("- 暂无关键风险发现。")
        lines.extend(["", "## 修复建议", ""])
        if payload["recommendations"]:
            for recommendation in payload["recommendations"]:
                lines.append(f"- {recommendation}")
        else:
            lines.append("- 暂无修复建议。")
        lines.extend(["", "## 分类得分", ""])
        for category, score in payload.get("category_scores", {}).items():
            lines.append(
                f"- `{category}`：{score['score']}，通过 {score['passed']}/{score['total_cases']}，pass_rate={score['pass_rate']}"
            )
        lines.extend(["", "## 攻击类型得分", ""])
        for attack_type, score in payload.get("attack_type_scores", {}).items():
            lines.append(
                f"- `{attack_type}`：{score['score']}，通过 {score['passed']}/{score['total_cases']}"
            )
        analysis = payload.get("failure_analysis", {})
        lines.extend(["", "## 失败原因聚合", ""])
        lines.append(f"- top_failed_attack_types：{analysis.get('top_failed_attack_types', [])}")
        lines.append(f"- top_false_positive_cases：{analysis.get('top_false_positive_cases', [])}")
        lines.append(f"- top_false_negative_cases：{analysis.get('top_false_negative_cases', [])}")
        lines.append(f"- high_confidence_failures：{analysis.get('high_confidence_failures', [])}")
        lines.extend(["", "## 前后对比指标", ""])
        lines.append(json.dumps(payload.get("comparison", {}), ensure_ascii=False, indent=2))
        lines.extend(["", "## 失败样例", ""])
        failed = [case for case in payload["cases"] if not case["passed"]]
        if failed:
            for case in failed:
                lines.append(f"### {case['case_id']}")
                lines.append("")
                lines.append(f"- 分数：{case['score']}")
                lines.append(f"- 期望：{case['metrics'].get('expected_behavior', '')}")
                lines.append("")
        else:
            lines.append("- 无失败样例。")
        return "\n".join(lines) + "\n"
