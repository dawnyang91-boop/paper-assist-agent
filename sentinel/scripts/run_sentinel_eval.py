from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sentinel.evaluators.runner import SentinelRunner
from sentinel.evaluators.target_adapters import build_adapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM-Sentinel evaluation.")
    parser.add_argument("--target", choices=["mock", "chatbot"], default="mock")
    parser.add_argument("--dataset", default="prompt_injection")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/chatbot")
    parser.add_argument("--output", default="sentinel/reports/sentinel_report")
    parser.add_argument("--api-token", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        dataset_path = root / "datasets" / f"{args.dataset}.yaml"
    adapter = build_adapter(args.target, base_url=args.base_url, api_token=args.api_token) if args.target == "chatbot" else build_adapter("mock")
    paths = SentinelRunner(adapter).run_and_report(dataset_path, args.output)
    print(f"JSON report: {paths['json']}")
    print(f"Markdown report: {paths['markdown']}")


if __name__ == "__main__":
    main()
