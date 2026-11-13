import subprocess

import pandas as pd
import pytest

from training.make_data import build
from training.provenance import dataset_provenance, dvc_md5
from training.train import main, train

MD5 = "d41d8cd98f00b204e9800998ecf8427e"


def git(cwd, *args):
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


def test_make_data_versions_differ_but_same_shape():
    v1, v2 = build("v1"), build("v2")
    assert v1.shape == v2.shape
    assert (v1["target"] != v2["target"]).mean() == pytest.approx(0.3, abs=0.01)
    assert v1.drop(columns="target").equals(v2.drop(columns="target"))
    with pytest.raises(ValueError):
        build("v3")


def test_degraded_data_gives_worse_f1(tmp_path):
    for version in ("v1", "v2"):
        build(version).to_csv(tmp_path / f"{version}.csv", index=False)
    _, good = train(tmp_path / "v1.csv")
    _, bad = train(tmp_path / "v2.csv")
    assert good["f1"] > 0.9
    assert bad["f1"] < good["f1"] - 0.1


def test_train_requires_label_column(tmp_path):
    pd.DataFrame({"a": [1, 2]}).to_csv(tmp_path / "x.csv", index=False)
    with pytest.raises(ValueError):
        train(tmp_path / "x.csv")


def test_cli_writes_metrics(tmp_path, capsys):
    build("v1").to_csv(tmp_path / "d.csv", index=False)
    out = tmp_path / "m.json"
    assert main(["--data", str(tmp_path / "d.csv"), "--metrics-out", str(out)]) == 0
    assert out.is_file() and "f1" in out.read_text()


def test_dvc_md5_reads_dvc_file(tmp_path):
    data = tmp_path / "data.csv"
    data.write_text("a\n1\n")
    assert dvc_md5(data) is None
    (tmp_path / "data.csv.dvc").write_text(f"outs:\n- md5: {MD5}\n  path: data.csv\n")
    assert dvc_md5(data) == MD5


def test_provenance_from_git_tag(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / "data.csv").write_text("a\n1\n")
    (tmp_path / "data.csv.dvc").write_text(f"outs:\n- md5: {MD5}\n  path: data.csv\n")
    git(tmp_path, "add", "data.csv.dvc")
    git(tmp_path, "commit", "-q", "-m", "data v1")
    git(tmp_path, "tag", "data-v1")
    tags = dataset_provenance("data.csv", tmp_path)
    assert tags["dvc.md5"] == MD5
    assert tags["dataset.version"] == "v1"
    assert tags["dvc.dataset_path"] == "data.csv"
    assert len(tags["git.commit"]) == 40
    assert "dvc.uncommitted_change" not in tags

    (tmp_path / "data.csv.dvc").write_text("outs:\n- md5: " + "a" * 32 + "\n")
    assert dataset_provenance("data.csv", tmp_path)["dvc.uncommitted_change"] == "true"


def test_provenance_omits_unknowns(tmp_path):
    (tmp_path / "data.csv").write_text("a\n1\n")
    tags = dataset_provenance("data.csv", tmp_path)
    assert tags == {"dvc.dataset_path": "data.csv"}
