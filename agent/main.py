"""Command-line chat with the MLOps agent.

python -m agent.main                      interactive chat
python -m agent.main --once "question"    one question, then exit
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from agent.config import load_config, load_dotenv
from agent.llm import LLMClient
from agent.logging_setup import setup_logging
from agent.orchestrator import AgentResult, Orchestrator
from agent.registry import build_registry
from agent.safety import ConfirmationManager


def format_result(result: AgentResult) -> str:
    lines = [result.answer]
    if result.tools:
        used = ", ".join(f"{t.name} [{'ok' if t.ok else 'failed'}]" for t in result.tools)
        lines += ["", f"Tools used: {used}"]
    lines.append(f"(request {result.request_id}, {result.status})")
    return "\n".join(lines)


def build_orchestrator() -> Orchestrator:
    config = load_config()
    if not (config.llm_api_key and config.llm_model):
        raise SystemExit(
            "Set LLM_API_KEY, LLM_MODEL (and LLM_BASE_URL) in .env or the environment."
        )
    confirmations = ConfirmationManager(config.confirm_ttl_s)
    registry = build_registry(config, confirmations)
    return Orchestrator(config, LLMClient(config), registry, confirmations)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", metavar="QUESTION", help="ask one question and exit")
    parser.add_argument("--log-file", default="logs/agent.jsonl", help="structured JSON logs")
    args = parser.parse_args(argv)

    load_dotenv()
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.log_file, "a", encoding="utf-8") as log_stream:
        setup_logging(logging.INFO, stream=log_stream)
        orchestrator = build_orchestrator()
        if args.once:
            result = orchestrator.chat(args.once)
            print(format_result(result))
            return 0 if result.status in ("ok", "rejected") else 1
        print("MLOps agent. Ask about datasets, experiments, the pipeline or the deployment.")
        print("Type 'exit' to quit. Reply 'confirm <token>' to approve a requested change.\n")
        while True:
            try:
                text = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if text.lower() in ("exit", "quit"):
                return 0
            if text:
                print("\n" + format_result(orchestrator.chat(text)) + "\n")


if __name__ == "__main__":
    sys.exit(main())
