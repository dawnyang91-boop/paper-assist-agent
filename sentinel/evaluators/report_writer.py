from __future__ import annotations

import json
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
            "category_scores": {},
            "key_findings": key_findings[:20],
            "recommendations": sorted(set(recommendations)),
            "cases": [result.to_dict() for result in results],
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
