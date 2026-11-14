"""Command-line chat with the MLOps agent.

python -m agent.main                      interactive chat
python -m agent.main --once "question"    one question, then exit
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

from agent.config import load_config, load_dotenv
from agent.llm import LLMClient
from agent.logging_setup import setup_logging
from agent.orchestrator import AgentResult, Orchestrator
from agent.registry import build_registry
from agent.safety import ConfirmationManager

TOKEN_RE = re.compile(r"confirmation_required.*?'([0-9a-f]{8})'", re.S)


def format_result(result: AgentResult, evidence: bool = False) -> str:
    lines = [result.answer]
    if result.tools:
        used = ", ".join(f"{t.name} [{'ok' if t.ok else 'failed'}]" for t in result.tools)
        lines += ["", f"Tools used: {used}"]
    if evidence:
        for t in result.tools:
            lines += ["", f"--- evidence: {t.name} args={json.dumps(t.arguments)}", t.output]
    lines.append(f"(request {result.request_id}, {result.status})")
    return "\n".join(lines)


def pending_token(result: AgentResult) -> str | None:
    """The confirmation token a tool asked for in this turn, read from the tool results."""
    for record in reversed(result.tools):
        match = TOKEN_RE.search((record.error or "") + record.output)
        if match:
            return match.group(1)
    return None


def build_orchestrator() -> Orchestrator:
    config = load_config()
    if not (config.llm_api_key and config.llm_model):
        raise SystemExit(
            "Set LLM_API_KEY, LLM_MODEL (and LLM_BASE_URL) in .env or the environment."
        )
    confirmations = ConfirmationManager(config.confirm_ttl_s)
    registry = build_registry(config, confirmations)
    return Orchestrator(config, LLMClient(config), registry, confirmations)


def run_script(orchestrator: Orchestrator, path: Path, evidence: bool) -> int:
    """Replay a conversation. Each line is typed by 'the user'; @confirm types the confirmation."""
    last: AgentResult | None = None
    status = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("@wait "):
            time.sleep(float(line.split()[1]))
            continue
        if line == "@confirm":
            token = pending_token(last) if last else None
            if not token:
                print("you> @confirm\n(no confirmation was requested)\n")
                return 1
            line = f"confirm {token}"
        print(f"you> {line}\n")
        last = orchestrator.chat(line)
        print(format_result(last, evidence) + "\n")
        if last.status not in ("ok", "rejected"):
            status = 1
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", metavar="QUESTION", help="ask one question and exit")
    parser.add_argument(
        "--script",
        metavar="FILE",
        help="run a conversation from a file: one user message per line; the line @confirm "
        "approves the token requested in the previous answer, as the user would by typing it; "
        "@wait N pauses N seconds",
    )
    parser.add_argument("--evidence", action="store_true", help="print each tool call and output")
    parser.add_argument("--log-file", default="logs/agent.jsonl", help="structured JSON logs")
    args = parser.parse_args(argv)

    load_dotenv()
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.log_file, "a", encoding="utf-8") as log_stream:
        setup_logging(logging.INFO, stream=log_stream)
        orchestrator = build_orchestrator()
        if args.once:
            result = orchestrator.chat(args.once)
            print(format_result(result, args.evidence))
            return 0 if result.status in ("ok", "rejected") else 1
        if args.script:
            return run_script(orchestrator, Path(args.script), args.evidence)
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
                print("\n" + format_result(orchestrator.chat(text), args.evidence) + "\n")


if __name__ == "__main__":
    sys.exit(main())
