"""DVC tools: dataset versions, provenance and reproduction.

All commands are fixed argument lists (no shell). DVC arguments pass through the allowlist
in `agent.safety`. Git is used read-only, except for the confirmed checkout tool.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from agent.config import Config
from agent.safety import ConfirmationManager, SafetyError, validate_dvc_args
from agent.schemas import ToolError, ToolSpec, object_schema

DVC_TIMEOUT_S = 90.0
_SKIP_DIRS = {".git", ".dvc", ".venv", "venv", "node_modules", "__pycache__", ".dvc-remote"}
_REV = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,99}$")
_PATH_PROP = {
    "type": "string",
    "description": "Dataset path, e.g. data/dataset.csv. Optional when only one dataset exists.",
}


def _posix_paths(value: Any) -> Any:
    """DVC prints Windows paths with backslashes; present them uniformly with slashes."""
    if isinstance(value, dict):
        return {_posix_paths(k): _posix_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_posix_paths(v) for v in value]
    if isinstance(value, str):
        return value.replace("\\", "/")
    return value


class DvcTools:
    def __init__(
        self, config: Config, confirmations: ConfirmationManager, timeout_s: float | None = None
    ):
        self.repo = Path(config.dvc_repo_path).resolve()
        self.confirmations = confirmations
        # DVC starts slowly (several seconds per command), so it gets a longer limit.
        self.timeout = timeout_s or max(config.tool_timeout_s, DVC_TIMEOUT_S)

    # ---- subprocess helpers -------------------------------------------------

    def _run(self, cmd: list[str]) -> str:
        try:
            done = subprocess.run(
                cmd, cwd=self.repo, capture_output=True, text=True, timeout=self.timeout
            )
        except subprocess.TimeoutExpired:
            raise ToolError(f"command timed out after {self.timeout}s") from None
        except OSError as exc:
            raise ToolError(f"cannot run command: {exc}") from None
        if done.returncode != 0:
            detail = (done.stderr or done.stdout).strip().splitlines()
            raise ToolError("command failed: " + (detail[-1] if detail else "no output"))
        return done.stdout

    def _dvc(self, args: list[str]) -> str:
        try:
            checked = validate_dvc_args(args)
        except SafetyError as exc:
            raise ToolError(str(exc)) from None
        return self._run([sys.executable, "-m", "dvc", *checked])

    def _git(self, args: list[str]) -> str:
        return self._run(["git", *args])

    def _rev(self, rev: str) -> str:
        if not isinstance(rev, str) or not _REV.match(rev) or ".." in rev:
            raise ToolError(f"invalid revision '{rev}'")
        try:
            return self._git(["rev-parse", "--verify", "--quiet", rev + "^{commit}"]).strip()
        except ToolError:
            raise ToolError(f"revision '{rev}' not found") from None

    # ---- dataset file helpers ----------------------------------------------

    def _dvc_files(self) -> list[Path]:
        found = []
        for p in self.repo.rglob("*.dvc"):
            rel = p.relative_to(self.repo)
            if not any(part in _SKIP_DIRS for part in rel.parts) and p.is_file():
                found.append(rel)
        return sorted(found)

    def _resolve(self, path: str | None) -> str:
        """Return the repo-relative posix path of a .dvc file."""
        files = [f.as_posix() for f in self._dvc_files()]
        if not path:
            if len(files) == 1:
                return files[0]
            raise ToolError(f"specify a dataset path; tracked: {files}")
        norm = PurePosixPath(path.replace("\\", "/"))
        if norm.is_absolute() or ".." in norm.parts or re.match(r"^[A-Za-z]:", path):
            raise ToolError("path must be relative and inside the project")
        name = norm.as_posix()
        if not name.endswith(".dvc"):
            name += ".dvc"
        if name not in files:
            raise ToolError(f"'{path}' is not tracked by DVC; tracked: {files}")
        return name

    @staticmethod
    def _parse(text: str) -> dict[str, Any]:
        try:
            doc = yaml.safe_load(text) or {}
            out = (doc.get("outs") or [{}])[0]
        except (yaml.YAMLError, AttributeError, IndexError, TypeError):
            raise ToolError("cannot parse .dvc file") from None
        return {"md5": out.get("md5"), "size": out.get("size"), "path": out.get("path")}

    def _at_rev(self, dvc_file: str, rev: str) -> dict[str, Any]:
        self._rev(rev)
        try:
            text = self._git(["show", f"{rev}:{dvc_file}"])
        except ToolError:
            raise ToolError(f"{dvc_file} does not exist at revision '{rev}'") from None
        return self._parse(text)

    def _last_change(self, dvc_file: str) -> dict[str, Any]:
        out = self._git(["log", "-1", "--format=%H%x1f%cI%x1f%s", "--", dvc_file]).strip()
        if not out:
            return {"commit": None, "date": None, "message": None, "tags": []}
        commit, date, message = out.split("\x1f", 2)
        tags = self._git(["tag", "--points-at", commit, "--list", "data-v*"]).split()
        return {"commit": commit, "date": date, "message": message, "tags": tags}

    # ---- tools --------------------------------------------------------------

    def list_datasets(self) -> dict[str, Any]:
        datasets = []
        for rel in self._dvc_files():
            info = self._parse((self.repo / rel).read_text(encoding="utf-8"))
            info["dvc_file"] = rel.as_posix()
            info["path"] = (rel.parent / (info["path"] or rel.stem)).as_posix()
            datasets.append(info)
        return {"datasets": datasets, "count": len(datasets)}

    def dataset_version(self, path: str | None = None) -> dict[str, Any]:
        dvc_file = self._resolve(path)
        current = self._parse((self.repo / dvc_file).read_text(encoding="utf-8"))
        change = self._last_change(dvc_file)
        dirty = bool(self._git(["status", "--porcelain", "--", dvc_file]).strip())
        tags = change["tags"]
        return {
            "dvc_file": dvc_file,
            "md5": current["md5"],
            "size": current["size"],
            "version": tags[0].removeprefix("data-") if tags else None,
            "version_tags": tags,
            "git_commit": change["commit"],
            "committed_at": change["date"],
            "commit_message": change["message"],
            "head_commit": self._git(["rev-parse", "HEAD"]).strip(),
            "uncommitted_change": dirty,
        }

    def compare_versions(self, rev_a: str, rev_b: str, path: str | None = None) -> dict[str, Any]:
        dvc_file = self._resolve(path)
        a, b = self._at_rev(dvc_file, rev_a), self._at_rev(dvc_file, rev_b)
        result: dict[str, Any] = {
            "dvc_file": dvc_file,
            rev_a: {"md5": a["md5"], "size": a["size"]},
            rev_b: {"md5": b["md5"], "size": b["size"]},
            "data_changed": a["md5"] != b["md5"],
        }
        try:
            diff = json.loads(self._dvc(["diff", rev_a, rev_b, "--json"]) or "{}")
            for items in diff.values():
                for item in items:
                    if isinstance(item, dict) and "path" in item:
                        item["path"] = str(item["path"]).replace("\\", "/")
            result["dvc_diff"] = diff
        except (ToolError, json.JSONDecodeError) as exc:
            result["dvc_diff_error"] = str(exc)
        return result

    def check_changes(self) -> dict[str, Any]:
        # Check each tracked dataset on its own: faster than a whole-project status, and it
        # does not mix in unrelated pipeline-stage staleness.
        status: dict[str, Any] = {}
        for dvc_file in self._dvc_files():
            try:
                part = json.loads(self._dvc(["status", dvc_file.as_posix(), "--json"]) or "{}")
            except json.JSONDecodeError:
                raise ToolError("could not parse dvc status output") from None
            status.update(part)
        # Say which version the files are on, so "unchanged" is never read as "same as an
        # older run's version".
        versions: dict[str, Any] = {}
        for dvc_file in self._dvc_files():
            info = self.dataset_version(dvc_file.as_posix())
            versions[dvc_file.as_posix()] = {"version": info["version"], "md5": info["md5"]}
        return {
            "changed": bool(status),
            "status": _posix_paths(status),
            "current_version": versions,
            "note": "'changed' compares files with the current version only. Compare "
            "current_version with the version recorded on a run before saying a run used "
            "the current data.",
        }

    def reproduce_instructions(self, rev: str, path: str | None = None) -> dict[str, Any]:
        dvc_file = self._resolve(path)
        meta = self._at_rev(dvc_file, rev)
        commit = self._rev(rev)
        data_path = (PurePosixPath(dvc_file).parent / (meta["path"] or "")).as_posix()
        return {
            "revision": rev,
            "commit": commit,
            "dvc_file": dvc_file,
            "expected_md5": meta["md5"],
            "steps": [
                f"git checkout {rev} -- {dvc_file}",
                f"dvc checkout {data_path}",
                f"md5sum {data_path}  # must equal {meta['md5']}",
                "dvc pull  # only if the data is not in the local cache",
            ],
            "note": "Steps are listed, not executed. Use dvc_checkout_version to run them.",
        }

    def checkout_version(
        self, rev: str, path: str | None = None, confirmation_token: str | None = None
    ) -> dict[str, Any]:
        dvc_file = self._resolve(path)
        meta = self._at_rev(dvc_file, rev)
        params = {"rev": rev, "dvc_file": dvc_file}
        if not confirmation_token:
            description = f"Check out {dvc_file} and its data at '{rev}' (md5 {meta['md5']})"
            token = self.confirmations.request("dvc_checkout_version", params, description)
            raise ToolError(
                "confirmation_required: ask the user to confirm token "
                f"'{token}' for: {description}. Then call again with confirmation_token."
            )
        if not self.confirmations.consume(confirmation_token, "dvc_checkout_version", params):
            raise ToolError("confirmation token is invalid, unapproved, expired or mismatched")
        if self._git(["status", "--porcelain", "--", dvc_file]).strip():
            raise ToolError(f"{dvc_file} has uncommitted changes; refusing to overwrite")
        self._git(["checkout", rev, "--", dvc_file])
        data_path = (PurePosixPath(dvc_file).parent / (meta["path"] or "")).as_posix()
        self._dvc(["checkout", data_path])
        return {"checked_out": rev, "dvc_file": dvc_file, "md5": meta["md5"]}

    # ---- registry -----------------------------------------------------------

    def specs(self) -> list[ToolSpec]:
        rev = {"type": "string", "description": "Git revision, tag (e.g. data-v1) or commit."}
        specs = [
            ToolSpec(
                "dvc_list_datasets",
                "List datasets tracked by DVC with their hash and size.",
                object_schema(),
                self.list_datasets,
            ),
            ToolSpec(
                "dvc_dataset_version",
                "Get the current version, hash and Git commit of a DVC-tracked dataset.",
                object_schema({"path": _PATH_PROP}),
                self.dataset_version,
            ),
            ToolSpec(
                "dvc_compare_versions",
                "Compare a dataset between two Git revisions: hashes and what changed.",
                object_schema({"rev_a": rev, "rev_b": rev, "path": _PATH_PROP}, ["rev_a", "rev_b"]),
                self.compare_versions,
            ),
            ToolSpec(
                "dvc_check_changes",
                "Check whether DVC-tracked data in the workspace differs from what is recorded.",
                object_schema(),
                self.check_changes,
            ),
            ToolSpec(
                "dvc_reproduce_instructions",
                "Explain how to reproduce the dataset of a past revision (not executed).",
                object_schema({"rev": rev, "path": _PATH_PROP}, ["rev"]),
                self.reproduce_instructions,
            ),
            ToolSpec(
                "dvc_checkout_version",
                "Restore the dataset of a past revision into the workspace. Changes local "
                "files, so it needs the user's confirmation.",
                object_schema(
                    {
                        "rev": rev,
                        "path": _PATH_PROP,
                        "confirmation_token": {
                            "type": "string",
                            "description": "Token the user approved. Omit on the first call.",
                        },
                    },
                    ["rev"],
                ),
                self.checkout_version,
                mutating=True,
            ),
        ]
        for spec in specs:
            spec.timeout_s = self.timeout + 5  # the orchestrator waits a little longer
        return specs
