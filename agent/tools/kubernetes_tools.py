"""Kubernetes tools using the official Python client.

Read tools inspect Deployments, Pods, Services, events and logs. Write tools (restart, scale)
are limited to an approved namespace and need the user's confirmation. The cluster side adds a
second layer: the agent's credentials come from a namespaced Role (see kubernetes/rbac.yaml).
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Callable
from typing import Any

from agent.config import Config
from agent.redact import redact, redact_obj
from agent.safety import ConfirmationManager, SafetyError, check_namespace
from agent.schemas import ToolError, ToolSpec, object_schema

_NAME = re.compile(r"^[a-z0-9]([-a-z0-9.]{0,251}[a-z0-9])?$")
MAX_LOG_LINES = 200
MAX_LOG_CHARS = 6000
BAD_WAITING_REASONS = {
    "CrashLoopBackOff",
    "ImagePullBackOff",
    "ErrImagePull",
    "CreateContainerConfigError",
    "CreateContainerError",
    "InvalidImageName",
    "RunContainerError",
}
VERSION_LABEL = "app.kubernetes.io/version"


def _check_name(value: str, what: str) -> str:
    if not isinstance(value, str) or not _NAME.match(value):
        raise ToolError(f"invalid {what}")
    return value


def _iso(value: Any) -> str | None:
    return value.isoformat() if value else None


def _age_s(value: Any, now: dt.datetime) -> int | None:
    return int((now - value).total_seconds()) if value else None


class KubernetesTools:
    def __init__(
        self,
        config: Config,
        confirmations: ConfirmationManager,
        core: Any = None,
        apps: Any = None,
        now: Callable[[], dt.datetime] | None = None,
    ):
        self.config = config
        self.confirmations = confirmations
        self._core = core
        self._apps = apps
        self._now = now or (lambda: dt.datetime.now(dt.timezone.utc))

    # ---- client plumbing ----------------------------------------------------

    def _load(self) -> None:
        from kubernetes import client, config

        try:
            config.load_kube_config(context=self.config.k8s_context or None)
        except Exception as exc:
            raise ToolError(f"cannot load kubeconfig: {type(exc).__name__}") from None
        self._core = client.CoreV1Api()
        self._apps = client.AppsV1Api()

    @property
    def core(self):
        if self._core is None:
            self._load()
        return self._core

    @property
    def apps(self):
        if self._apps is None:
            self._load()
        return self._apps

    def _call(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, _request_timeout=self.config.tool_timeout_s, **kwargs)
        except ToolError:
            raise
        except Exception as exc:
            status = getattr(exc, "status", None)
            if status == 403:
                raise ToolError(
                    "forbidden: the agent's Kubernetes role does not allow this"
                ) from None
            if status == 404:
                raise ToolError("not found in Kubernetes") from None
            if status:
                raise ToolError(f"Kubernetes API error {status}") from None
            raise ToolError(f"Kubernetes API unavailable: {type(exc).__name__}") from None

    def _ns(self, namespace: str | None) -> str:
        ns = namespace or self.config.k8s_allowed_namespaces[0]
        try:
            return check_namespace(ns, self.config.k8s_allowed_namespaces)
        except SafetyError as exc:
            raise ToolError(str(exc)) from None

    # ---- summaries ----------------------------------------------------------

    def _pod_summary(self, pod: Any) -> dict[str, Any]:
        statuses = pod.status.container_statuses or []
        ready = sum(1 for c in statuses if c.ready)
        waiting = [
            c.state.waiting.reason
            for c in statuses
            if c.state and c.state.waiting and c.state.waiting.reason
        ]
        labels = pod.metadata.labels or {}
        return {
            "name": pod.metadata.name,
            "phase": pod.status.phase,
            "ready": f"{ready}/{len(pod.spec.containers)}",
            "restarts": sum(c.restart_count for c in statuses),
            "waiting_reasons": waiting,
            "node": pod.spec.node_name,
            "age_s": _age_s(pod.metadata.creation_timestamp, self._now()),
            "version": labels.get(VERSION_LABEL),
            "images": [c.image for c in pod.spec.containers],
        }

    @staticmethod
    def _state(state: Any) -> dict[str, Any] | None:
        if state is None:
            return None
        if state.waiting:
            return {
                "state": "waiting",
                "reason": state.waiting.reason,
                "message": redact(state.waiting.message or "")[:300],
            }
        if state.terminated:
            t = state.terminated
            return {
                "state": "terminated",
                "reason": t.reason,
                "exit_code": t.exit_code,
                "message": redact(t.message or "")[:300],
                "finished_at": _iso(t.finished_at),
            }
        if state.running:
            return {"state": "running", "started_at": _iso(state.running.started_at)}
        return None

    # ---- tools --------------------------------------------------------------

    def list_deployments(self, namespace: str | None = None) -> dict[str, Any]:
        ns = self._ns(namespace)
        items = self._call(self.apps.list_namespaced_deployment, ns).items
        out = []
        for d in items:
            st = d.status
            out.append(
                {
                    "name": d.metadata.name,
                    "desired": d.spec.replicas,
                    "ready": st.ready_replicas or 0,
                    "available": st.available_replicas or 0,
                    "updated": st.updated_replicas or 0,
                    "images": [c.image for c in d.spec.template.spec.containers],
                    "version": (d.metadata.labels or {}).get(VERSION_LABEL),
                    "conditions": [
                        {"type": c.type, "status": c.status, "reason": c.reason}
                        for c in (st.conditions or [])
                    ],
                }
            )
        return {"namespace": ns, "deployments": out}

    def list_pods(self, namespace: str | None = None) -> dict[str, Any]:
        ns = self._ns(namespace)
        pods = self._call(self.core.list_namespaced_pod, ns).items
        return {"namespace": ns, "pods": [self._pod_summary(p) for p in pods]}

    def pod_status(self, pod: str, namespace: str | None = None) -> dict[str, Any]:
        ns = self._ns(namespace)
        _check_name(pod, "pod name")
        p = self._call(self.core.read_namespaced_pod, pod, ns)
        containers = []
        specs = {c.name: c for c in p.spec.containers}
        for c in p.status.container_statuses or []:
            spec = specs.get(c.name)

            def probe(pr):
                if pr is None:
                    return None
                http = pr.http_get
                return {"path": http.path, "port": str(http.port)} if http else {"type": "other"}

            containers.append(
                {
                    "name": c.name,
                    "ready": c.ready,
                    "restart_count": c.restart_count,
                    "image": c.image,
                    "state": self._state(c.state),
                    "last_state": self._state(c.last_state),
                    "config_sources": [
                        {"configmap": ref.config_map_ref.name}
                        for ref in (spec.env_from or [])
                        if ref.config_map_ref
                    ]
                    if spec
                    else [],
                    "readiness_probe": probe(spec.readiness_probe) if spec else None,
                    "liveness_probe": probe(spec.liveness_probe) if spec else None,
                }
            )
        return {
            **self._pod_summary(p),
            "conditions": [
                {"type": c.type, "status": c.status, "reason": c.reason, "message": c.message}
                for c in (p.status.conditions or [])
            ],
            "containers": containers,
        }

    def pod_logs(
        self, pod: str, namespace: str | None = None, tail_lines: int = 50, previous: bool = False
    ) -> dict[str, Any]:
        ns = self._ns(namespace)
        _check_name(pod, "pod name")
        tail = max(1, min(int(tail_lines), MAX_LOG_LINES))
        text = self._call(
            self.core.read_namespaced_pod_log, pod, ns, tail_lines=tail, previous=bool(previous)
        )
        text = redact(text or "")
        return {
            "pod": pod,
            "previous": bool(previous),
            "tail_lines": tail,
            "logs": text[-MAX_LOG_CHARS:],
            "truncated": len(text) > MAX_LOG_CHARS,
        }

    def list_services(self, namespace: str | None = None) -> dict[str, Any]:
        ns = self._ns(namespace)
        services = self._call(self.core.list_namespaced_service, ns).items
        out = []
        for s in services:
            ready = not_ready = 0
            try:
                ep = self._call(self.core.read_namespaced_endpoints, s.metadata.name, ns)
                for sub in ep.subsets or []:
                    ready += len(sub.addresses or [])
                    not_ready += len(sub.not_ready_addresses or [])
            except ToolError:
                ready = not_ready = -1  # unknown
            out.append(
                {
                    "name": s.metadata.name,
                    "type": s.spec.type,
                    "cluster_ip": s.spec.cluster_ip,
                    "ports": [
                        {"port": p.port, "target_port": str(p.target_port), "protocol": p.protocol}
                        for p in s.spec.ports or []
                    ],
                    "selector": s.spec.selector,
                    "ready_endpoints": ready,
                    "not_ready_endpoints": not_ready,
                }
            )
        return {"namespace": ns, "services": out}

    def events(
        self, namespace: str | None = None, object_name: str | None = None
    ) -> dict[str, Any]:
        ns = self._ns(namespace)
        if object_name:
            _check_name(object_name, "object name")
        items = self._call(self.core.list_namespaced_event, ns).items
        rows = []
        for e in items:
            obj = e.involved_object.name if e.involved_object else None
            if object_name and obj != object_name:
                continue
            when = e.last_timestamp or e.event_time or e.metadata.creation_timestamp
            rows.append(
                {
                    "time": _iso(when),
                    "type": e.type,
                    "reason": e.reason,
                    "object": obj,
                    "message": redact(e.message or "")[:300],
                    "count": e.count,
                }
            )
        rows.sort(key=lambda r: r["time"] or "", reverse=True)
        return {
            "namespace": ns,
            "events": rows[:30],
            "warnings": sum(1 for r in rows if r["type"] == "Warning"),
        }

    def configmaps(self, namespace: str | None = None, name: str | None = None) -> dict[str, Any]:
        """Non-sensitive configuration. Secrets are never read."""
        ns = self._ns(namespace)
        if name:
            _check_name(name, "configmap name")
        items = self._call(self.core.list_namespaced_config_map, ns).items
        out = [
            {"name": c.metadata.name, "data": redact_obj(dict(c.data or {}))}
            for c in items
            if not name or c.metadata.name == name
        ]
        if name and not out:
            raise ToolError("not found in Kubernetes")
        return {"namespace": ns, "configmaps": out}

    def unhealthy_pods(self, namespace: str | None = None) -> dict[str, Any]:
        ns = self._ns(namespace)
        pods = self._call(self.core.list_namespaced_pod, ns).items
        unhealthy = []
        for p in pods:
            summary = self._pod_summary(p)
            reasons = []
            if p.status.phase not in ("Running", "Succeeded"):
                reasons.append(f"phase {p.status.phase}")
            for c in p.status.container_statuses or []:
                if not c.ready:
                    reasons.append(f"container {c.name} not ready")
                if c.state and c.state.waiting and c.state.waiting.reason in BAD_WAITING_REASONS:
                    reasons.append(f"waiting: {c.state.waiting.reason}")
                if c.restart_count:
                    last = self._state(c.last_state)
                    detail = f"{c.restart_count} restart(s)"
                    if last and last["state"] == "terminated":
                        detail += f", last exit {last['exit_code']} ({last['reason']})"
                    reasons.append(detail)
            if reasons:
                unhealthy.append({**summary, "problems": reasons})
        return {
            "namespace": ns,
            "checked": len(pods),
            "unhealthy": unhealthy,
            "all_healthy": not unhealthy,
        }

    def deployed_version(
        self, deployment: str | None = None, namespace: str | None = None
    ) -> dict[str, Any]:
        ns = self._ns(namespace)
        deployments = self._call(self.apps.list_namespaced_deployment, ns).items
        if deployment:
            _check_name(deployment, "deployment name")
            deployments = [d for d in deployments if d.metadata.name == deployment]
        if not deployments:
            raise ToolError("no matching deployment found")
        results = []
        for d in deployments:
            images = [c.image for c in d.spec.template.spec.containers]
            entry: dict[str, Any] = {
                "deployment": d.metadata.name,
                "images": images,
                "image_tags": [
                    i.rsplit(":", 1)[1] if ":" in i.rsplit("/", 1)[-1] else "latest" for i in images
                ],
                "version_label": (d.metadata.labels or {}).get(VERSION_LABEL),
                "source": "deployment spec and labels",
            }
            svc = next(iter(self._services_for(d, ns)), None)
            if svc:
                try:
                    live = self._call(
                        self.core.connect_get_namespaced_service_proxy_with_path,
                        f"{svc}:http",
                        ns,
                        path="version",
                    )
                    entry["live_endpoint_version"] = live if isinstance(live, str) else str(live)
                except ToolError as exc:
                    entry["live_endpoint_version"] = f"unavailable ({exc})"
            results.append(entry)
        return {"namespace": ns, "deployments": results}

    def _services_for(self, deployment: Any, ns: str) -> list[str]:
        labels = deployment.spec.selector.match_labels or {}
        try:
            services = self._call(self.core.list_namespaced_service, ns).items
        except ToolError:
            return []
        return [
            s.metadata.name
            for s in services
            if s.spec.selector and all(labels.get(k) == v for k, v in s.spec.selector.items())
        ]

    # ---- state-changing tools ----------------------------------------------

    def _confirmed(self, action: str, params: dict, description: str, token: str | None) -> None:
        if not token:
            new = self.confirmations.request(action, params, description)
            raise ToolError(
                f"confirmation_required: ask the user to confirm token '{new}' for: "
                f"{description}. Then call again with confirmation_token."
            )
        if not self.confirmations.consume(token, action, params):
            raise ToolError("confirmation token is invalid, unapproved, expired or mismatched")

    def _existing_deployment(self, name: str, ns: str) -> Any:
        _check_name(name, "deployment name")
        return self._call(self.apps.read_namespaced_deployment, name, ns)

    def restart_deployment(
        self, deployment: str, namespace: str | None = None, confirmation_token: str | None = None
    ) -> dict[str, Any]:
        ns = self._ns(namespace)
        self._existing_deployment(deployment, ns)
        params = {"deployment": deployment, "namespace": ns}
        self._confirmed(
            "k8s_restart_deployment",
            params,
            f"Rolling restart of deployment '{deployment}' in '{ns}'",
            confirmation_token,
        )
        stamp = self._now().isoformat()
        body = {
            "spec": {
                "template": {
                    "metadata": {"annotations": {"kubectl.kubernetes.io/restartedAt": stamp}}
                }
            }
        }
        self._call(self.apps.patch_namespaced_deployment, deployment, ns, body)
        return {"restarted": deployment, "namespace": ns, "restarted_at": stamp}

    def scale_deployment(
        self,
        deployment: str,
        replicas: int,
        namespace: str | None = None,
        confirmation_token: str | None = None,
    ) -> dict[str, Any]:
        ns = self._ns(namespace)
        if not isinstance(replicas, int) or isinstance(replicas, bool) or not 1 <= replicas <= 5:
            raise ToolError("replicas must be an integer from 1 to 5")
        current = self._existing_deployment(deployment, ns)
        params = {"deployment": deployment, "namespace": ns, "replicas": replicas}
        self._confirmed(
            "k8s_scale_deployment",
            params,
            f"Scale '{deployment}' in '{ns}' from {current.spec.replicas} to {replicas}",
            confirmation_token,
        )
        self._call(
            self.apps.patch_namespaced_deployment_scale,
            deployment,
            ns,
            {"spec": {"replicas": replicas}},
        )
        return {
            "scaled": deployment,
            "namespace": ns,
            "from": current.spec.replicas,
            "to": replicas,
        }

    # ---- registry -----------------------------------------------------------

    def specs(self) -> list[ToolSpec]:
        ns = {"type": "string", "description": "Namespace. Omit for the default approved one."}
        pod = {"type": "string", "description": "Pod name."}
        dep = {"type": "string", "description": "Deployment name."}
        token = {"type": "string", "description": "Token the user approved. Omit on first call."}
        return [
            ToolSpec(
                "k8s_list_deployments",
                "List Deployments with replica counts, images and version.",
                object_schema({"namespace": ns}),
                self.list_deployments,
            ),
            ToolSpec(
                "k8s_list_pods",
                "List Pods with phase, readiness, restarts and images.",
                object_schema({"namespace": ns}),
                self.list_pods,
            ),
            ToolSpec(
                "k8s_pod_status",
                "Detailed Pod status: conditions, container states, last termination, probes.",
                object_schema({"pod": pod, "namespace": ns}, ["pod"]),
                self.pod_status,
            ),
            ToolSpec(
                "k8s_pod_logs",
                "Get recent Pod logs (redacted). Use previous=true for a crashed container.",
                object_schema(
                    {
                        "pod": pod,
                        "namespace": ns,
                        "tail_lines": {"type": "integer", "minimum": 1, "maximum": 200},
                        "previous": {"type": "boolean"},
                    },
                    ["pod"],
                ),
                self.pod_logs,
            ),
            ToolSpec(
                "k8s_list_services",
                "Inspect Services: type, ports, selector and ready endpoints.",
                object_schema({"namespace": ns}),
                self.list_services,
            ),
            ToolSpec(
                "k8s_events",
                "Recent namespace events, optionally for one object.",
                object_schema({"namespace": ns, "object_name": {"type": "string"}}),
                self.events,
            ),
            ToolSpec(
                "k8s_get_configmaps",
                "Read ConfigMaps (non-sensitive settings such as model path or versions) to check "
                "how a workload is configured.",
                object_schema({"namespace": ns, "name": {"type": "string"}}),
                self.configmaps,
            ),
            ToolSpec(
                "k8s_unhealthy_pods",
                "Find unhealthy Pods and restart events with reasons.",
                object_schema({"namespace": ns}),
                self.unhealthy_pods,
            ),
            ToolSpec(
                "k8s_deployed_version",
                "Report the deployed application version from image "
                "tag, version label and the live /version endpoint.",
                object_schema({"namespace": ns, "deployment": dep}),
                self.deployed_version,
            ),
            ToolSpec(
                "k8s_restart_deployment",
                "Rolling restart of a Deployment. Changes cluster "
                "state: needs the user's confirmation and an approved namespace.",
                object_schema(
                    {"deployment": dep, "namespace": ns, "confirmation_token": token},
                    ["deployment"],
                ),
                self.restart_deployment,
                mutating=True,
            ),
            ToolSpec(
                "k8s_scale_deployment",
                "Scale a Deployment to 1-5 replicas. Changes cluster "
                "state: needs the user's confirmation and an approved namespace.",
                object_schema(
                    {
                        "deployment": dep,
                        "replicas": {"type": "integer", "minimum": 1, "maximum": 5},
                        "namespace": ns,
                        "confirmation_token": token,
                    },
                    ["deployment", "replicas"],
                ),
                self.scale_deployment,
                mutating=True,
            ),
        ]
