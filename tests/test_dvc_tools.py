import os
import shutil
import subprocess
import sys

import pytest

pytest.importorskip("dvc")

from agent.config import Config  # noqa: E402
from agent.safety import ConfirmationManager  # noqa: E402
from agent.schemas import ToolError  # noqa: E402
from agent.tools.dvc_tools import DvcTools  # noqa: E402

GIT_ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.com",
    "DVC_NO_ANALYTICS": "true",
}


def run(cwd, *cmd):
    env = {**os.environ, **GIT_ENV}
    return subprocess.run(cmd, cwd=cwd, env=env, check=True, capture_output=True, text=True)


def dvc(cwd, *args):
    return run(cwd, sys.executable, "-m", "dvc", *args)


@pytest.fixture(scope="module")
def base_repo(tmp_path_factory):
    root = tmp_path_factory.mktemp("proj")
    run(root, "git", "init", "-q", "-b", "main")
    dvc(root, "init", "-q")
    (root / "data").mkdir()
    for version, text in (("v1", "a,b\n1,2\n"), ("v2", "a,b\n1,2\n3,4\n")):
        (root / "data" / "d.csv").write_text(text)
        dvc(root, "add", "data/d.csv")
        run(root, "git", "add", "-A")
        run(root, "git", "commit", "-q", "-m", f"data {version}")
        run(root, "git", "tag", f"data-{version}")
    return root


@pytest.fixture
def repo(base_repo, tmp_path):
    copy = tmp_path / "proj"
    shutil.copytree(base_repo, copy)
    return copy


def make_tools(root, ttl=300):
    cm = ConfirmationManager(ttl)
    return DvcTools(Config(dvc_repo_path=str(root)), cm, timeout_s=60), cm


def md5_of(root, rev):
    out = run(root, "git", "show", f"{rev}:data/d.csv.dvc").stdout
    return next(line.split(":")[1].strip() for line in out.splitlines() if "md5" in line)


def test_list_datasets(repo):
    tools, _ = make_tools(repo)
    result = tools.list_datasets()
    assert result["count"] == 1
    assert result["datasets"][0]["path"] == "data/d.csv"
    assert result["datasets"][0]["md5"] == md5_of(repo, "data-v2")


def test_dataset_version_reports_tag_and_commit(repo):
    tools, _ = make_tools(repo)
    info = tools.dataset_version()
    commit = run(repo, "git", "rev-parse", "data-v2").stdout.strip()
    assert info["version"] == "v2"
    assert info["git_commit"] == commit
    assert info["md5"] == md5_of(repo, "data-v2")
    assert info["uncommitted_change"] is False
    assert tools.dataset_version("data/d.csv")["version"] == "v2"


def test_compare_versions(repo):
    tools, _ = make_tools(repo)
    diff = tools.compare_versions("data-v1", "data-v2")
    assert diff["data_changed"] is True
    assert diff["data-v1"]["md5"] != diff["data-v2"]["md5"]
    assert diff["dvc_diff"]["modified"][0]["path"] == "data/d.csv"
    same = tools.compare_versions("data-v2", "data-v2")
    assert same["data_changed"] is False


def test_check_changes_detects_modification(repo):
    tools, _ = make_tools(repo)
    assert tools.check_changes()["changed"] is False
    (repo / "data" / "d.csv").write_text("a,b\n9,9\n")
    result = tools.check_changes()
    assert result["changed"] is True
    assert "data/d.csv.dvc" in result["status"]


def test_reproduce_instructions_contain_expected_hash(repo):
    tools, _ = make_tools(repo)
    out = tools.reproduce_instructions("data-v1")
    assert out["expected_md5"] == md5_of(repo, "data-v1")
    assert any("dvc checkout data/d.csv" in s for s in out["steps"])
    assert any("git checkout data-v1 -- data/d.csv.dvc" in s for s in out["steps"])


def test_checkout_requires_confirmation_then_restores_v1(repo):
    tools, cm = make_tools(repo)
    with pytest.raises(ToolError, match="confirmation_required") as err:
        tools.checkout_version("data-v1")
    token = str(err.value).split("'")[1]
    # unapproved token is rejected, and the data is untouched
    with pytest.raises(ToolError, match="invalid"):
        tools.checkout_version("data-v1", confirmation_token=token)
    assert (repo / "data" / "d.csv").read_text() == "a,b\n1,2\n3,4\n"
    assert cm.approve(token)
    result = tools.checkout_version("data-v1", confirmation_token=token)
    assert result["md5"] == md5_of(repo, "data-v1")
    assert (repo / "data" / "d.csv").read_text() == "a,b\n1,2\n"
    with pytest.raises(ToolError):  # token is single use
        tools.checkout_version("data-v1", confirmation_token=token)


def test_checkout_refuses_forged_token(repo):
    tools, _ = make_tools(repo)
    with pytest.raises(ToolError, match="invalid"):
        tools.checkout_version("data-v1", confirmation_token="deadbeef")


@pytest.mark.parametrize(
    "call",
    [
        lambda t: t.compare_versions("data-v1; whoami", "data-v2"),
        lambda t: t.compare_versions("--output=x", "data-v2"),
        lambda t: t.compare_versions("data-v1", "no-such-rev"),
        lambda t: t.dataset_version("../outside"),
        lambda t: t.dataset_version("/etc/passwd"),
        lambda t: t.dataset_version("data/missing.csv"),
        lambda t: t.reproduce_instructions("a..b"),
    ],
)
def test_invalid_requests_raise_tool_error(repo, call):
    tools, _ = make_tools(repo)
    with pytest.raises(ToolError):
        call(tools)


def test_specs_expose_schemas_and_flag_mutation(repo):
    tools, _ = make_tools(repo)
    specs = {s.name: s for s in tools.specs()}
    assert len(specs) >= 3
    assert specs["dvc_checkout_version"].mutating is True
    assert all(not s.mutating for n, s in specs.items() if n != "dvc_checkout_version")
    for spec in specs.values():
        assert spec.to_openai()["function"]["parameters"]["type"] == "object"
