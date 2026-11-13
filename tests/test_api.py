import joblib
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from api.main import create_app  # noqa: E402
from training.make_data import build  # noqa: E402
from training.train import train  # noqa: E402


@pytest.fixture(scope="module")
def model_file(tmp_path_factory):
    root = tmp_path_factory.mktemp("model")
    csv = root / "d.csv"
    frame = build("v1")
    frame.to_csv(csv, index=False)
    model, _ = train(csv)
    path = root / "model.joblib"
    joblib.dump(model, path)
    return path, frame


def test_ready_and_predict_when_model_loaded(model_file, monkeypatch):
    path, frame = model_file
    monkeypatch.setenv("MODEL_VERSION", "v1")
    monkeypatch.setenv("APP_VERSION", "9.9.9")
    with TestClient(create_app(str(path))) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").status_code == 200
        version = client.get("/version").json()
        assert version == {"app_version": "9.9.9", "model_version": "v1", "model_loaded": True}
        row = frame.drop(columns="target").iloc[0].tolist()
        out = client.post("/predict", json={"features": row}).json()
        assert out["prediction"] in (0, 1) and 0 <= out["probability"] <= 1
        assert out["model_version"] == "v1"


def test_predict_validates_feature_count(model_file):
    path, _ = model_file
    with TestClient(create_app(str(path))) as client:
        assert client.post("/predict", json={"features": [1.0, 2.0]}).status_code == 422
        assert client.post("/predict", json={"features": []}).status_code == 422
        assert client.post("/predict", json={"nope": 1}).status_code == 422


def test_missing_model_is_alive_but_not_ready(tmp_path):
    with TestClient(create_app(str(tmp_path / "absent.joblib"))) as client:
        assert client.get("/health").status_code == 200  # liveness must not restart the pod
        assert client.get("/ready").status_code == 503
        assert client.get("/version").json()["model_loaded"] is False
        assert client.post("/predict", json={"features": [1.0]}).status_code == 503


def test_manifests_meet_the_spec():
    import pathlib

    import yaml

    root = pathlib.Path(__file__).resolve().parents[1] / "kubernetes"
    docs = {p.stem: yaml.safe_load(p.read_text(encoding="utf-8")) for p in root.glob("*.yaml")}
    dep = docs["deployment"]
    assert dep["kind"] == "Deployment" and dep["spec"]["replicas"] >= 2
    container = dep["spec"]["template"]["spec"]["containers"][0]
    assert "readinessProbe" in container and "livenessProbe" in container
    for section in ("requests", "limits"):
        assert {"cpu", "memory"} <= set(container["resources"][section])
    assert container["envFrom"][0]["configMapRef"]["name"] == docs["configmap"]["metadata"]["name"]
    assert not container["image"].endswith(":latest")
    assert docs["service"]["spec"]["type"] == "ClusterIP"
    assert docs["configmap"]["kind"] == "ConfigMap"
    assert "secret" not in str(docs["configmap"]).lower()
    labels = dep["metadata"]["labels"]
    assert labels["app.kubernetes.io/version"] == docs["configmap"]["data"]["APP_VERSION"]
