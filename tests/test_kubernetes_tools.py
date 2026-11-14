import datetime as dt
import json

import pytest

pytest.importorskip("kubernetes")

from agent.config import Config  # noqa: E402
from agent.safety import ConfirmationManager  # noqa: E402
from agent.schemas import ToolError  # noqa: E402
from agent.tools.kubernetes_tools import KubernetesTools  # noqa: E402
from kubernetes import client as k  # noqa: E402

NOW = dt.datetime(2026, 11, 14, 12, 0, tzinfo=dt.timezone.utc)
NS = "mlops-agent"
SECRET = "sk-" + "q1w2e3r4" * 4


class ApiError(Exception):
    def __init__(self, status):
        self.status = status


def pod(
    name,
    *,
    phase="Running",
    ready=True,
    restarts=0,
    waiting=None,
    last_exit=None,
    image="img:1.0.0",
):
    state = k.V1ContainerState(running=k.V1ContainerStateRunning(started_at=NOW))
    if waiting:
        state = k.V1ContainerState(
            waiting=k.V1ContainerStateWaiting(reason=waiting, message="boom")
        )
    last = None
    if last_exit is not None:
        last = k.V1ContainerState(
            terminated=k.V1ContainerStateTerminated(
                exit_code=last_exit, reason="Error", finished_at=NOW
            )
        )
    status = k.V1ContainerStatus(
        name="api",
        image=image,
        image_id="",
        ready=ready,
        restart_count=restarts,
        state=state,
        last_state=last,
    )
    container = k.V1Container(
        name="api",
        image=image,
        readiness_probe=k.V1Probe(http_get=k.V1HTTPGetAction(path="/ready", port="http")),
        liveness_probe=k.V1Probe(http_get=k.V1HTTPGetAction(path="/health", port="http")),
    )
    return k.V1Pod(
        metadata=k.V1ObjectMeta(
            name=name,
            labels={"app.kubernetes.io/version": "1.0.0"},
            creation_timestamp=NOW - dt.timedelta(minutes=5),
        ),
        spec=k.V1PodSpec(containers=[container], node_name="node-1"),
        status=k.V1PodStatus(
            phase=phase,
            container_statuses=[status],
            conditions=[
                k.V1PodCondition(
                    type="Ready",
                    status="True" if ready else "False",
                    reason=None if ready else "ContainersNotReady",
                )
            ],
        ),
    )


def deployment(name="inference-api", replicas=2, ready=2, image="img:1.0.0"):
    return k.V1Deployment(
        metadata=k.V1ObjectMeta(name=name, labels={"app.kubernetes.io/version": "1.0.0"}),
        spec=k.V1DeploymentSpec(
            replicas=replicas,
            selector=k.V1LabelSelector(match_labels={"app": "api"}),
            template=k.V1PodTemplateSpec(
                spec=k.V1PodSpec(containers=[k.V1Container(name="api", image=image)])
            ),
        ),
        status=k.V1DeploymentStatus(
            ready_replicas=ready,
            available_replicas=ready,
            updated_replicas=replicas,
            conditions=[
                k.V1DeploymentCondition(
                    type="Available", status="True", reason="MinimumReplicasAvailable"
                )
            ],
        ),
    )


class FakeCore:
    def __init__(self):
        self.pods = [pod("good-1"), pod("good-2")]
        self.logs = "line1\nline2"
        self.fail = None
        self.calls = []

    def _maybe_fail(self):
        if self.fail:
            raise self.fail

    def list_namespaced_pod(self, ns, **kw):
        self.calls.append(("list_pod", ns))
        self._maybe_fail()
        return k.V1PodList(items=self.pods)

    def read_namespaced_pod(self, name, ns, **kw):
        self._maybe_fail()
        found = [p for p in self.pods if p.metadata.name == name]
        if not found:
            raise ApiError(404)
        return found[0]

    def read_namespaced_pod_log(self, name, ns, **kw):
        self.calls.append(("logs", kw))
        self._maybe_fail()
        return self.logs

    def list_namespaced_config_map(self, ns, **kw):
        return k.V1ConfigMapList(
            items=[
                k.V1ConfigMap(
                    metadata=k.V1ObjectMeta(name="inference-api-config"),
                    data={"MODEL_PATH": "/app/models/model.joblib", "API_TOKEN": "t0k3n-value"},
                )
            ]
        )

    def list_namespaced_service(self, ns, **kw):
        return k.V1ServiceList(
            items=[
                k.V1Service(
                    metadata=k.V1ObjectMeta(name="inference-api"),
                    spec=k.V1ServiceSpec(
                        type="ClusterIP",
                        cluster_ip="10.0.0.1",
                        selector={"app": "api"},
                        ports=[k.V1ServicePort(port=80, target_port="http", protocol="TCP")],
                    ),
                )
            ]
        )

    def read_namespaced_endpoints(self, name, ns, **kw):
        return k.V1Endpoints(
            subsets=[
                k.V1EndpointSubset(
                    addresses=[k.V1EndpointAddress(ip="1.1.1.1")],
                    not_ready_addresses=[k.V1EndpointAddress(ip="1.1.1.2")],
                )
            ]
        )

    def list_namespaced_event(self, ns, **kw):
        def ev(reason, type_, obj, minutes, msg):
            return k.CoreV1Event(
                metadata=k.V1ObjectMeta(name=reason + obj, creation_timestamp=NOW),
                involved_object=k.V1ObjectReference(name=obj),
                reason=reason,
                type=type_,
                message=msg,
                last_timestamp=NOW - dt.timedelta(minutes=minutes),
                count=1,
            )

        return k.CoreV1EventList(
            items=[
                ev("Pulled", "Normal", "good-1", 10, "pulled"),
                ev("Unhealthy", "Warning", "bad-1", 1, f"Readiness probe failed token={SECRET}"),
            ]
        )

    def connect_get_namespaced_service_proxy_with_path(self, name, ns, path, **kw):
        self._maybe_fail()
        return '{"app_version": "1.0.0"}'


class FakeApps:
    def __init__(self):
        self.deployments = [deployment()]
        self.patches = []

    def list_namespaced_deployment(self, ns, **kw):
        return k.V1DeploymentList(items=self.deployments)

    def read_namespaced_deployment(self, name, ns, **kw):
        found = [d for d in self.deployments if d.metadata.name == name]
        if not found:
            raise ApiError(404)
        return found[0]

    def patch_namespaced_deployment(self, name, ns, body, **kw):
        self.patches.append(("restart", name, ns, body))

    def patch_namespaced_deployment_scale(self, name, ns, body, **kw):
        self.patches.append(("scale", name, ns, body))


@pytest.fixture
def env():
    core, apps, cm = FakeCore(), FakeApps(), ConfirmationManager()
    cfg = Config(k8s_allowed_namespaces=(NS,))
    return KubernetesTools(cfg, cm, core, apps, now=lambda: NOW), core, apps, cm


def test_list_deployments_and_pods(env):
    tools, *_ = env
    dep = tools.list_deployments()["deployments"][0]
    assert (dep["desired"], dep["ready"], dep["version"]) == (2, 2, "1.0.0")
    pods = tools.list_pods()["pods"]
    assert [p["name"] for p in pods] == ["good-1", "good-2"]
    assert pods[0]["ready"] == "1/1" and pods[0]["age_s"] == 300


def test_pod_status_has_probes_and_state(env):
    tools, core, *_ = env
    core.pods = [pod("bad-1", ready=False, restarts=4, waiting="CrashLoopBackOff", last_exit=1)]
    st = tools.pod_status("bad-1")
    c = st["containers"][0]
    assert c["state"]["reason"] == "CrashLoopBackOff"
    assert c["last_state"]["exit_code"] == 1
    assert c["readiness_probe"] == {"path": "/ready", "port": "http"}
    assert st["conditions"][0]["reason"] == "ContainersNotReady"


def test_unhealthy_pods_finds_crashloop_and_image_pull(env):
    tools, core, *_ = env
    core.pods = [
        pod("ok"),
        pod("crash", ready=False, restarts=5, waiting="CrashLoopBackOff", last_exit=137),
        pod("pull", phase="Pending", ready=False, waiting="ImagePullBackOff"),
    ]
    out = tools.unhealthy_pods()
    names = {p["name"]: p["problems"] for p in out["unhealthy"]}
    assert set(names) == {"crash", "pull"} and out["all_healthy"] is False
    assert any("last exit 137" in x for x in names["crash"])
    assert any("ImagePullBackOff" in x for x in names["pull"])


def test_all_healthy(env):
    tools, *_ = env
    assert tools.unhealthy_pods()["all_healthy"] is True


def test_logs_are_redacted_and_bounded(env):
    tools, core, *_ = env
    core.logs = f"started\nauth failed token={SECRET}\n" + "x" * 10000
    out = tools.pod_logs("good-1", tail_lines=9999)
    assert SECRET not in out["logs"] and out["truncated"] is True
    assert out["tail_lines"] == 200 and len(out["logs"]) <= 6000


def test_services_report_endpoints(env):
    tools, *_ = env
    svc = tools.list_services()["services"][0]
    assert svc["type"] == "ClusterIP" and svc["ready_endpoints"] == 1
    assert svc["not_ready_endpoints"] == 1 and svc["ports"][0]["port"] == 80


def test_events_sorted_filtered_and_redacted(env):
    tools, *_ = env
    out = tools.events()
    assert out["events"][0]["reason"] == "Unhealthy" and out["warnings"] == 1
    assert SECRET not in json.dumps(out)
    assert [e["object"] for e in tools.events(object_name="good-1")["events"]] == ["good-1"]


def test_deployed_version_combines_spec_and_live_endpoint(env):
    tools, core, *_ = env
    out = tools.deployed_version()["deployments"][0]
    assert out["image_tags"] == ["1.0.0"] and out["version_label"] == "1.0.0"
    assert "1.0.0" in out["live_endpoint_version"]
    core.fail = ApiError(503)
    assert "unavailable" in tools.deployed_version()["deployments"][0]["live_endpoint_version"]


def test_image_without_tag_is_reported_as_latest(env):
    tools, _, apps, _ = env
    apps.deployments = [deployment(image="registry.local:5000/app")]
    assert tools.deployed_version()["deployments"][0]["image_tags"] == ["latest"]


@pytest.mark.parametrize("ns", ["kube-system", "default", "other"])
def test_namespace_outside_allowlist_is_rejected_without_api_calls(env, ns):
    tools, core, apps, _ = env
    for call in (
        lambda: tools.list_pods(ns),
        lambda: tools.pod_logs("good-1", ns),
        lambda: tools.restart_deployment("inference-api", ns),
        lambda: tools.scale_deployment("inference-api", 3, ns),
    ):
        with pytest.raises(ToolError, match="not approved"):
            call()
    assert core.calls == [] and apps.patches == []


def test_restart_needs_confirmation_and_changes_nothing_first(env):
    tools, _, apps, cm = env
    with pytest.raises(ToolError, match="confirmation_required") as err:
        tools.restart_deployment("inference-api")
    token = str(err.value).split("'")[1]
    assert apps.patches == []
    with pytest.raises(ToolError, match="invalid"):
        tools.restart_deployment("inference-api", confirmation_token=token)  # not approved
    cm.approve(token)
    out = tools.restart_deployment("inference-api", confirmation_token=token)
    assert out["restarted"] == "inference-api"
    kind, name, ns, body = apps.patches[0]
    assert (kind, name, ns) == ("restart", "inference-api", NS)
    assert "restartedAt" in json.dumps(body)
    with pytest.raises(ToolError):  # single use
        tools.restart_deployment("inference-api", confirmation_token=token)
    assert len(apps.patches) == 1


def test_forged_or_mismatched_token_is_rejected(env):
    tools, _, apps, cm = env
    with pytest.raises(ToolError, match="invalid"):
        tools.restart_deployment("inference-api", confirmation_token="deadbeef")
    with pytest.raises(ToolError) as err:
        tools.scale_deployment("inference-api", 3)
    token = str(err.value).split("'")[1]
    cm.approve(token)
    with pytest.raises(ToolError, match="invalid"):  # token was issued for 3 replicas, not 4
        tools.scale_deployment("inference-api", 4, confirmation_token=token)
    with pytest.raises(ToolError, match="invalid"):  # or for restart
        tools.restart_deployment("inference-api", confirmation_token=token)
    assert apps.patches == []


def test_scale_validates_and_applies(env):
    tools, _, apps, cm = env
    for bad in (0, 6, -1, "3", True, 2.5):
        with pytest.raises(ToolError, match="replicas"):
            tools.scale_deployment("inference-api", bad)
    with pytest.raises(ToolError) as err:
        tools.scale_deployment("inference-api", 3)
    token = str(err.value).split("'")[1]
    cm.approve(token)
    out = tools.scale_deployment("inference-api", 3, confirmation_token=token)
    assert (out["from"], out["to"]) == (2, 3)
    assert apps.patches[0][3] == {"spec": {"replicas": 3}}


def test_unknown_deployment_and_invalid_names(env):
    tools, *_ = env
    with pytest.raises(ToolError, match="not found"):
        tools.restart_deployment("ghost")
    for bad in ("Bad_Name", "x;y", "../a", ""):
        with pytest.raises(ToolError, match="invalid"):
            tools.pod_status(bad)


def test_api_errors_are_mapped(env):
    tools, core, *_ = env
    core.fail = ApiError(403)
    with pytest.raises(ToolError, match="forbidden"):
        tools.list_pods()
    core.fail = ApiError(500)
    with pytest.raises(ToolError, match="API error 500"):
        tools.list_pods()
    core.fail = ConnectionError("refused")
    with pytest.raises(ToolError, match="unavailable"):
        tools.list_pods()


def test_specs_flag_only_write_tools_as_mutating(env):
    tools, *_ = env
    specs = {s.name: s for s in tools.specs()}
    assert len(specs) >= 3
    assert {n for n, s in specs.items() if s.mutating} == {
        "k8s_restart_deployment",
        "k8s_scale_deployment",
    }


def test_configmaps_are_readable_and_sensitive_keys_masked(env):
    tools, *_ = env
    out = tools.configmaps()["configmaps"][0]
    assert out["data"]["MODEL_PATH"] == "/app/models/model.joblib"
    assert out["data"]["API_TOKEN"] == "[REDACTED]"
    assert len(tools.configmaps(name="inference-api-config")["configmaps"]) == 1
    with pytest.raises(ToolError, match="not found"):
        tools.configmaps(name="nope")
    with pytest.raises(ToolError, match="not approved"):
        tools.configmaps("kube-system")


def test_pod_status_names_the_config_source(env):
    tools, core, *_ = env
    p = pod("cfg-1")
    p.spec.containers[0].env_from = [
        k.V1EnvFromSource(config_map_ref=k.V1ConfigMapEnvSource(name="inference-api-config"))
    ]
    core.pods = [p]
    sources = tools.pod_status("cfg-1")["containers"][0]["config_sources"]
    assert sources == [{"configmap": "inference-api-config"}]
