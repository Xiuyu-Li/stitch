"""SGLang engine request construction and version-stamping tests."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from stitch.engines.base import EngineHealthStatus
from stitch.engines.sglang import SGLangEngine
from stitch.sync import AdmissionGate
from stitch.types import VersionKind, VersionManifest, VersionRef


def _manifest(kind: VersionKind = VersionKind.DELTA) -> VersionManifest:
    return VersionManifest(VersionRef("r1", 5), kind, ["weights"])


def _engine(mode: str = "disk") -> SGLangEngine:
    return SGLangEngine(
        "http://engine",
        "/ckpt" if mode == "disk" else None,
        delta_update_mode=mode,  # type: ignore[arg-type]
    )


def test_stamp_request_namespaces_by_version() -> None:
    engine = _engine()
    req: dict = {"text": "hi"}
    engine.stamp_request(req, VersionRef("r1", 7))
    assert req["extra_key"] == "wv7;r1/"
    listed: dict = {"extra_key": ["a", "b"]}
    engine.stamp_request(listed, VersionRef(None, 3))
    assert listed["extra_key"] == ["wv3;a", "wv3;b"]


def test_delta_update_mode_is_validated() -> None:
    with pytest.raises(ValueError, match="delta_update_mode"):
        _engine("memory")


def test_disk_mode_requires_local_checkpoint() -> None:
    with pytest.raises(ValueError, match="requires local_checkpoint_dir"):
        SGLangEngine("http://engine")


def test_cpu_mode_does_not_require_local_checkpoint() -> None:
    _engine("cpu")


@pytest.mark.parametrize("mode", ["disk", "cpu"])
def test_initialize_verifies_sglang_startup_contract(mode: str) -> None:
    engine = _engine(mode)
    requests: list[str] = []

    async def fake_get(path, *, ok=(200,)):
        del ok
        requests.append(path)
        if path == "/server_info":
            return {
                "weight_update_staging": mode,
                "weight_update_local_checkpoint_dir": (
                    "/ckpt" if mode == "disk" else None
                ),
            }
        return {"weight_version": "119"}

    engine._get_json = fake_get  # type: ignore[method-assign]
    asyncio.run(engine.initialize_update_destination(119))
    assert requests == ["/server_info", "/model_info"]


@pytest.mark.parametrize(
    ("server_info", "model_info", "message"),
    [
        (
            {
                "weight_update_staging": "disk",
                "weight_update_local_checkpoint_dir": None,
            },
            {"weight_version": "3"},
            "staging mode",
        ),
        (
            {
                "weight_update_staging": "cpu",
                "weight_update_local_checkpoint_dir": None,
            },
            {"weight_version": "2"},
            "startup weight version",
        ),
    ],
)
def test_initialize_rejects_mismatched_sglang_contract(
    server_info: dict, model_info: dict, message: str
) -> None:
    engine = _engine("cpu")

    async def fake_get(path, *, ok=(200,)):
        del ok
        return server_info if path == "/server_info" else model_info

    engine._get_json = fake_get  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match=message):
        asyncio.run(engine.initialize_update_destination(3))


def test_initialize_rejects_mismatched_checkpoint_directory() -> None:
    engine = _engine("disk")

    async def fake_get(path, *, ok=(200,)):
        del ok
        if path == "/server_info":
            return {
                "weight_update_staging": "disk",
                "weight_update_local_checkpoint_dir": "/other",
            }
        return {"weight_version": "0"}

    engine._get_json = fake_get  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="checkpoint directory"):
        asyncio.run(engine.initialize_update_destination())


@pytest.mark.parametrize("mode", ["disk", "cpu"])
def test_stage_prepares_one_target(mode: str) -> None:
    engine = _engine(mode)
    requests = []

    async def fake_post(path, payload, *, timeout=None, action=None):
        requests.append((path, payload, timeout, action))

    engine._post = fake_post  # type: ignore[method-assign]
    asyncio.run(engine.stage(_manifest(), "/source/weight_v000005"))
    assert requests == [
        (
            "/prepare_weight_update",
            {"checkpoint_source_dir": "/source", "target_version": 5},
            3600.0,
            "weight preparation",
        )
    ]


@pytest.mark.parametrize("mode", ["disk", "cpu"])
def test_commit_publishes_the_prepared_target(mode: str) -> None:
    engine = _engine(mode)
    requests = []

    async def fake_post(path, payload, *, timeout=None, action=None):
        requests.append((path, payload, timeout, action))

    engine._post = fake_post  # type: ignore[method-assign]
    asyncio.run(engine.commit(_manifest(), flush_cache=False))
    assert requests == [
        (
            "/commit_weight_update",
            {
                "target_version": 5,
                "abort_all_requests": False,
                "torch_empty_cache": False,
                "flush_cache": False,
            },
            600.0,
            "weight commit",
        )
    ]


def test_cpu_mode_rejects_full_checkpoint() -> None:
    engine = _engine("cpu")
    with pytest.raises(ValueError, match="delta manifests only"):
        asyncio.run(engine.stage(_manifest(VersionKind.FULL), "/source/weight_v000005"))


@pytest.mark.parametrize("mode", ["disk", "cpu"])
def test_reset_requires_a_fresh_replica(mode: str) -> None:
    with pytest.raises(RuntimeError, match="fresh rollout replica"):
        asyncio.run(_engine(mode).reset())


def test_stamp_response_generate_vs_openai() -> None:
    engine = _engine()
    gen: dict = {"text": "x", "meta_info": {}}
    engine.stamp_response(gen, VersionRef("r1", 4), VersionRef("r1", 5))
    assert gen["meta_info"] == {
        "weight_version": "4",
        "weight_version_start": 4,
        "weight_version_end": 5,
    }
    openai: dict = {"choices": [{"meta_info": {}}]}
    engine.stamp_response(openai, VersionRef("r1", 4), VersionRef("r1", 4))
    assert openai["weight_version_start"] == 4
    assert openai["weight_version_end"] == 4
    assert openai["choices"][0]["meta_info"] == {
        "weight_version": "4",
        "weight_version_start": 4,
        "weight_version_end": 4,
    }
    assert "meta_info" not in openai and "weight_version" not in openai


class _HealthClient:
    def __init__(self, outcome) -> None:
        self.outcome = outcome
        self.urls: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args) -> None:
        pass

    async def get(self, url: str) -> httpx.Response:
        self.urls.append(url)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return httpx.Response(self.outcome, request=httpx.Request("GET", url))


@pytest.mark.parametrize(
    "outcome,expected",
    [
        (200, EngineHealthStatus.HEALTHY),
        (503, EngineHealthStatus.UNRESPONSIVE),
        (
            httpx.ReadTimeout("busy", request=httpx.Request("GET", "http://engine")),
            EngineHealthStatus.UNRESPONSIVE,
        ),
        (
            httpx.ConnectError(
                "connection refused",
                request=httpx.Request("GET", "http://engine"),
            ),
            EngineHealthStatus.UNREACHABLE,
        ),
    ],
)
def test_health_check_classifies_engine_failures(
    monkeypatch, outcome, expected
) -> None:
    client = _HealthClient(outcome)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kwargs: client)
    engine = _engine()
    assert asyncio.run(engine.check_health()).status is expected
    assert client.urls == ["http://engine/health"]


@pytest.mark.parametrize("failure", [400, 404, "body", "timeout", "cancelled"])
def test_failed_cache_drain_releases_health_guard(monkeypatch, failure) -> None:
    async def go() -> None:
        async def handle(request):
            if request.url.path == "/flush_cache":
                assert request.url.params["timeout"] == "0.25"
                assert request.extensions["timeout"]["read"] > 0.25
                if failure == "cancelled":
                    raise asyncio.CancelledError
                if failure == "timeout":
                    raise httpx.ReadTimeout("drain timed out", request=request)
                return httpx.Response(
                    200 if failure == "body" else failure,
                    json={"success": False, "message": "not idle"},
                )
            return httpx.Response(200)

        client = httpx.AsyncClient
        transport = httpx.MockTransport(handle)
        monkeypatch.setattr(
            httpx, "AsyncClient", lambda **kw: client(transport=transport, **kw)
        )
        engine = SGLangEngine("http://engine", "/ckpt", control_timeout=0.25)
        expected = {
            "timeout": httpx.ReadTimeout,
            "cancelled": asyncio.CancelledError,
        }.get(failure, RuntimeError)
        with pytest.raises(expected):
            async with engine.commit_guard(flush_cache=True):
                pytest.fail("a failed drain must prevent the weight change")
        async with asyncio.timeout(1):
            assert (await engine.check_health()).status is EngineHealthStatus.HEALTHY
            async with engine.commit_guard():
                pass

    asyncio.run(go())


@pytest.mark.parametrize("failure", ["pause", "apply", "resume"])
@pytest.mark.parametrize("cancelled", [False, True])
def test_failed_commit_releases_health_guard_and_admission(
    monkeypatch, failure, cancelled
) -> None:
    async def go() -> None:
        failed_path = {
            "pause": "/pause_generation",
            "apply": "/commit_weight_update",
            "resume": "/continue_generation",
        }[failure]
        reached_failure = asyncio.Event()
        calls, applied = [], []

        async def handle(request):
            calls.append(request.url.path)
            if request.url.path == failed_path:
                if cancelled:
                    reached_failure.set()
                    await asyncio.Event().wait()
                return httpx.Response(500, json={"message": "control RPC failed"})
            return httpx.Response(200, json={"success": True})

        client = httpx.AsyncClient
        transport = httpx.MockTransport(handle)
        monkeypatch.setattr(
            httpx, "AsyncClient", lambda **kw: client(transport=transport, **kw)
        )
        engine = SGLangEngine("http://engine", delta_update_mode="cpu")
        gate = AdmissionGate(served_version=lambda: VersionRef("r1", 0))
        async with asyncio.timeout(1):
            task = asyncio.create_task(
                gate.commit(
                    apply=lambda: engine.commit(_manifest(VersionKind.DELTA)),
                    on_applied=lambda: applied.append(True),
                    pause=engine.pause,
                    resume=engine.resume,
                    guard=engine.commit_guard,
                )
            )
            if cancelled:
                await reached_failure.wait()
                task.cancel()
            with pytest.raises(asyncio.CancelledError if cancelled else RuntimeError):
                await task
            # Resume still runs after an interrupted apply; failures cannot strand
            # the guard or gate. Reconciler's existing terminal-error policy stays.
            if failure == "apply":
                assert "/continue_generation" in calls
            assert bool(applied) == (failure == "resume")
            assert (await engine.check_health()).status is EngineHealthStatus.HEALTHY
            async with gate.admit():
                pass

    asyncio.run(go())


def test_cancelled_guard_waiter_does_not_unlock_an_active_probe(monkeypatch) -> None:
    async def go() -> None:
        started, finish = asyncio.Event(), asyncio.Event()
        calls = []

        async def handle(request):
            calls.append(request.url.path)
            started.set()
            await finish.wait()
            return httpx.Response(200)

        client = httpx.AsyncClient
        transport = httpx.MockTransport(handle)
        monkeypatch.setattr(
            httpx, "AsyncClient", lambda **kw: client(transport=transport, **kw)
        )
        engine = SGLangEngine("http://engine", "/ckpt")

        async def update():
            async with engine.commit_guard(flush_cache=True):
                pytest.fail("the active probe still owns the guard")

        async with asyncio.timeout(1), asyncio.TaskGroup() as tasks:
            first_probe = tasks.create_task(engine.check_health())
            await started.wait()
            waiter = tasks.create_task(update())
            await asyncio.sleep(0)
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            second_probe = tasks.create_task(engine.check_health())
            await asyncio.sleep(0)
            assert calls == ["/health"] and not second_probe.done()
            finish.set()
            assert (await first_probe).status is EngineHealthStatus.HEALTHY
            assert (await second_probe).status is EngineHealthStatus.HEALTHY

    asyncio.run(go())
