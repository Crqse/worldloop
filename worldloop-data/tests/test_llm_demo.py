"""Tests for ``scripts/run_llm_demo.py`` (B3-07).

Covers the delivery contract:
- fake mode end-to-end produces every contract artifact;
- inference_events ↔ transitions alignment by (episode, tick, agent)
  and by ``_inference_id``;
- fallback counting is correct when the client picks illegal actions;
- a fake API key placed in the environment never appears in any output;
- real mode fails loudly on a missing key (no silent fake fallback) and
  wires ``OpenAICompatibleClient`` when the key is present (stubbed
  here — no network call in tests);
- the quality chain (exporter → leakage → quality reporter) runs by
  default, producing quality_report.json / leakage_report.json with
  actually-judged Q items (not all skipped);
- real mode fail-closed: a run whose decisions are ALL fallback (no
  actual LLM response entered the data) exits non-zero.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pytest

from worldloop_data.llm_policy import (
    EchoLLMClient,
    LLMRequest,
    LLMResponse,
    LLMServerError,
)

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_llm_demo.py"

# A recognizable fake secret; the leak-scan test asserts it never lands
# in any artifact.
_FAKE_KEY = "fake-api-key-test-FAKE-SECRET-0xDEADBEEF-do-not-leak"
_KEY_ENV = "WORLDLOOP_LLM_API_KEY"

_CONTRACT_FILES = (
    "manifest.md",
    "resolved_config.json",
    "transitions.jsonl",
    "inference_events.jsonl",
    "summary.json",
    "README.md",
)

# Written by the quality chain (default-on --with-quality).
_QUALITY_FILES = (
    "quality_report.json",
    "leakage_report.json",
    "coverage_report.json",
)


def _load_demo_module(name: str = "run_llm_demo_under_test"):
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Shared fake-mode run (2 seeds × 4 ticks) with a fake key in the env
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def demo_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    mod = _load_demo_module("run_llm_demo_shared")
    out = tmp_path_factory.mktemp("llm_demo") / "llm_demo_fake"
    saved = os.environ.get(_KEY_ENV)
    os.environ[_KEY_ENV] = _FAKE_KEY
    try:
        rc = mod.main(
            [
                "--mode",
                "fake",
                "--seeds",
                "42,43",
                "--ticks",
                "4",
                "--output",
                str(out),
            ]
        )
    finally:
        if saved is None:
            os.environ.pop(_KEY_ENV, None)
        else:
            os.environ[_KEY_ENV] = saved
    assert rc == 0
    return out


# ---------------------------------------------------------------------------
# 1. Contract files
# ---------------------------------------------------------------------------


class TestContractFiles:
    def test_all_contract_files_exist(self, demo_run: Path) -> None:
        for name in _CONTRACT_FILES:
            assert (demo_run / name).exists(), f"missing contract file: {name}"

    def test_episode_dirs_with_raw_records(self, demo_run: Path) -> None:
        for episode_id in ("seed42_run0", "seed43_run1"):
            ep_dir = demo_run / "episodes" / episode_id
            assert ep_dir.is_dir()
            assert (ep_dir / "manifest.json").exists()
            assert len(list(ep_dir.glob("t*.json"))) == 4

    def test_resolved_config_records_key_name_only(self, demo_run: Path) -> None:
        cfg = json.loads((demo_run / "resolved_config.json").read_text("utf-8"))
        assert cfg["policy"]["api_key_env"] == _KEY_ENV
        assert cfg["mode"] == "fake"
        assert cfg["seeds"] == [42, 43]


# ---------------------------------------------------------------------------
# 2. Summary metrics
# ---------------------------------------------------------------------------


class TestSummary:
    def test_required_metric_keys(self, demo_run: Path) -> None:
        summary = json.loads((demo_run / "summary.json").read_text("utf-8"))
        for key in (
            "proposals_total",
            "world_accepted",
            "accept_rate",
            "parse_failures",
            "illegal_candidates",
            "fallback_count",
            "api_errors",
            "total_tokens",
            "wall_time_seconds",
            "latency_ms_p50",
            "latency_ms_p95",
            "transitions_recorded",
        ):
            assert key in summary, f"summary.json missing {key}"

    def test_metric_values_consistent(self, demo_run: Path) -> None:
        summary = json.loads((demo_run / "summary.json").read_text("utf-8"))
        # FakeLLMClient always proposes REST, which is always legal in
        # emergency_resource → no fallback, everything accepted.
        assert summary["proposals_total"] == 8  # 2 seeds × 4 ticks
        assert summary["transitions_recorded"] == 8
        assert summary["world_accepted"] == 8
        assert summary["accept_rate"] == 1.0
        assert summary["fallback_count"] == 0
        assert summary["parse_failures"] == 0
        assert summary["illegal_candidates"] == 0
        assert summary["api_errors"] == 0
        assert summary["total_tokens"] > 0
        assert "synthetic" in summary["total_tokens_note"]


# ---------------------------------------------------------------------------
# 3. Alignment: inference_events ↔ transitions
# ---------------------------------------------------------------------------


class TestAlignment:
    def test_events_align_with_transitions(self, demo_run: Path) -> None:
        transitions = _read_jsonl(demo_run / "transitions.jsonl")
        events = _read_jsonl(demo_run / "inference_events.jsonl")
        assert len(transitions) == 8
        assert len(events) == 8

        by_key = {(e["episode_id"], e["tick"], e["agent_id"]): e for e in events}
        by_id = {e["inference_id"]: e for e in events}

        for rec in transitions:
            episode_id = rec["_episode_id"]
            tick = rec["tick"]
            for agent_id, cand in rec["candidate_actions"].items():
                key = (episode_id, tick, str(agent_id))
                assert key in by_key, f"no inference event for {key}"
                event = by_key[key]
                # Non-fallback proposals carry the inference id in params.
                inf_id = cand["params"].get("_inference_id")
                assert inf_id in by_id
                assert by_id[inf_id] is event or by_id[inf_id] == event
                # world_accepted back-filled from the receipt.
                receipt = rec["receipts"][agent_id]
                assert event["world_accepted"] == bool(receipt["success"])


# ---------------------------------------------------------------------------
# 4. Fallback counting (illegal candidate → first_legal fallback)
# ---------------------------------------------------------------------------


class TestFallback:
    def test_fallback_counted_when_candidate_illegal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mod = _load_demo_module("run_llm_demo_fallback")
        # The demo builds its fake client via the module-level
        # FakeLLMClient name; swap in a client that always proposes an
        # action type absent from the scenario's action set.
        monkeypatch.setattr(
            mod, "FakeLLMClient", lambda: EchoLLMClient("FLY_TO_MOON")
        )
        out = tmp_path / "fallback_run"
        rc = mod.main(
            ["--mode", "fake", "--seeds", "42", "--ticks", "3", "--output", str(out)]
        )
        assert rc == 0

        summary = json.loads((out / "summary.json").read_text("utf-8"))
        assert summary["transitions_recorded"] == 3
        assert summary["fallback_count"] == 3
        assert summary["illegal_candidates"] == 3
        assert summary["proposals_total"] == 3  # fallback still proposes

        events = _read_jsonl(out / "inference_events.jsonl")
        assert all(e["fallback_used"] for e in events)
        assert all(e["error_type"] == "illegal_action" for e in events)


# ---------------------------------------------------------------------------
# 5. Secret hygiene: env key never appears in any artifact
# ---------------------------------------------------------------------------


class TestSecretHygiene:
    def test_fake_key_absent_from_all_outputs(self, demo_run: Path) -> None:
        scanned = 0
        for path in demo_run.rglob("*"):
            if not path.is_file():
                continue
            content = path.read_text(encoding="utf-8", errors="ignore")
            assert _FAKE_KEY not in content, f"API key leaked into {path}"
            scanned += 1
        assert scanned >= len(_CONTRACT_FILES)


# ---------------------------------------------------------------------------
# 6. Real mode: loud failure, never silent fake
# ---------------------------------------------------------------------------


class TestRealMode:
    def test_missing_key_errors_out(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mod = _load_demo_module("run_llm_demo_real_nokey")
        missing_env = "B3_07_TEST_KEY_THAT_DOES_NOT_EXIST"
        monkeypatch.delenv(missing_env, raising=False)
        out = tmp_path / "real_nokey"
        with pytest.raises(SystemExit) as excinfo:
            mod.main(
                [
                    "--mode",
                    "real",
                    "--base-url",
                    "http://localhost:9999/v1",
                    "--model",
                    "test-model",
                    "--api-key-env",
                    missing_env,
                    "--output",
                    str(out),
                ]
            )
        # Non-zero exit with a message naming the env var — not the value.
        assert excinfo.value.code
        assert missing_env in str(excinfo.value.code)
        # Nothing was produced: no silent fallback to fake mode.
        assert not (out / "summary.json").exists()

    def test_present_key_wires_real_client(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With a key present, real mode constructs OpenAICompatibleClient
        (stubbed — no network) and completes a run labelled mode=real."""
        mod = _load_demo_module("run_llm_demo_real_key")
        env_name = "B3_07_TEST_REAL_KEY"
        monkeypatch.setenv(env_name, _FAKE_KEY)

        constructed: List[Dict[str, Any]] = []

        class _StubRealClient:
            def __init__(self, base_url: str, api_key_env: str, timeout_seconds: float) -> None:
                constructed.append(
                    {
                        "base_url": base_url,
                        "api_key_env": api_key_env,
                        "timeout_seconds": timeout_seconds,
                    }
                )

            def complete(self, request: LLMRequest) -> LLMResponse:
                body = {"action_type": "REST", "params": {}, "reason_code": "STUB"}
                return LLMResponse(
                    raw_text=json.dumps(body),
                    json_body=body,
                    finish_reason="stop",
                    input_tokens=100,
                    output_tokens=10,
                )

        monkeypatch.setattr(mod, "OpenAICompatibleClient", _StubRealClient)
        out = tmp_path / "real_key"
        rc = mod.main(
            [
                "--mode",
                "real",
                "--base-url",
                "http://localhost:9999/v1",
                "--model",
                "test-model",
                "--api-key-env",
                env_name,
                "--seeds",
                "42",
                "--ticks",
                "2",
                "--output",
                str(out),
            ]
        )
        assert rc == 0
        # One client per episode, wired from CLI args.
        assert constructed == [
            {
                "base_url": "http://localhost:9999/v1",
                "api_key_env": env_name,
                # Real-mode auto timeout (reasoning-style endpoints).
                "timeout_seconds": 120.0,
            }
        ]
        summary = json.loads((out / "summary.json").read_text("utf-8"))
        assert summary["mode"] == "real"
        assert summary["transitions_recorded"] == 2
        assert summary["total_tokens"] == 220  # (100+10) × 2 ticks, from usage
        cfg = json.loads((out / "resolved_config.json").read_text("utf-8"))
        assert cfg["policy"]["client"] == "OpenAICompatibleClient"
        assert cfg["policy"]["api_key_env"] == env_name
        # Key never leaks into any artifact.
        for path in out.rglob("*"):
            if path.is_file():
                assert _FAKE_KEY not in path.read_text(
                    encoding="utf-8", errors="ignore"
                ), f"API key leaked into {path}"

    def test_all_fallback_real_run_fails_closed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """mode=real with a client that never yields a usable response
        must exit non-zero (evidence fail-closed — the data would be
        100% fallback, i.e. effectively mock mislabelled as real)."""
        mod = _load_demo_module("run_llm_demo_real_allfallback")
        env_name = "B3_07_TEST_FAILCLOSED_KEY"
        monkeypatch.setenv(env_name, _FAKE_KEY)

        class _DeadClient:
            def __init__(self, base_url: str, api_key_env: str, timeout_seconds: float) -> None:
                pass

            def complete(self, request: LLMRequest) -> LLMResponse:
                raise LLMServerError("stub 503: endpoint unavailable")

        monkeypatch.setattr(mod, "OpenAICompatibleClient", _DeadClient)
        out = tmp_path / "real_all_fallback"
        with pytest.raises(SystemExit) as excinfo:
            mod.main(
                [
                    "--mode",
                    "real",
                    "--base-url",
                    "http://localhost:9999/v1",
                    "--model",
                    "test-model",
                    "--api-key-env",
                    env_name,
                    "--seeds",
                    "42",
                    "--ticks",
                    "2",
                    "--output",
                    str(out),
                ]
            )
        msg = str(excinfo.value.code)
        assert excinfo.value.code, "expected a non-zero exit"
        assert "fail-closed" in msg
        # Error message never carries key material.
        assert _FAKE_KEY not in msg
        # Artifacts are kept for diagnosis (no silent success).
        assert (out / "summary.json").exists()


# ---------------------------------------------------------------------------
# 7. Quality chain: exporter → leakage → quality reporter
# ---------------------------------------------------------------------------


class TestQualityChain:
    def test_quality_files_exist(self, demo_run: Path) -> None:
        for name in _QUALITY_FILES:
            assert (demo_run / name).exists(), f"missing quality artifact: {name}"
        # Published dataset from the exporter step.
        assert (demo_run / "dataset" / "manifest.json").exists()
        assert (demo_run / "dataset" / "transitions.jsonl").exists()

    def test_quality_report_items_actually_judged(self, demo_run: Path) -> None:
        report = json.loads((demo_run / "quality_report.json").read_text("utf-8"))
        items = report["items"]
        assert len(items) >= 10
        keys = {it["key"] for it in items}
        assert {f"Q{i}" for i in range(10)} <= keys
        for it in items:
            assert it["status"] in ("pass", "fail", "skipped"), it
        # The gate must actually judge the data — not report all-skipped.
        judged = [it for it in items if it["status"] in ("pass", "fail")]
        assert len(judged) >= 5, f"quality gate barely judged anything: {items}"
        # Core data-integrity gates are judgeable on fake-mode data and
        # must hold on it (Q0 schema, Q1 traceability, Q5 leakage, Q8
        # quarantine identity). A failure here is a wiring regression.
        by_key = {it["key"]: it for it in items}
        for key in ("Q0", "Q1", "Q5", "Q8"):
            assert by_key[key]["status"] == "pass", by_key[key]

    def test_leakage_report_shape(self, demo_run: Path) -> None:
        leakage = json.loads((demo_run / "leakage_report.json").read_text("utf-8"))
        assert leakage["ok"] is True
        assert leakage["violation_count"] == 0
        assert "seed" in leakage["checked_kinds"]

    def test_summary_carries_quality_block(self, demo_run: Path) -> None:
        summary = json.loads((demo_run / "summary.json").read_text("utf-8"))
        quality = summary["quality"]
        report = json.loads((demo_run / "quality_report.json").read_text("utf-8"))
        assert quality["overall"] == report["overall"]
        assert quality["passed"] == report["passed"]
        assert quality["failed"] == report["failed"]
        assert quality["skipped"] == report["skipped"]
        assert quality["leakage_ok"] is True
        # Q8 quantity identity surfaces in the export block.
        exp = quality["export"]
        assert exp["produced"] == (
            exp["accepted"] + exp["quarantined"] + exp["explicitly_rejected"]
        )
        assert exp["dropped"] == 0

    def test_provenance_carries_inference_config(self, demo_run: Path) -> None:
        """Q4 support: rollout stamps the policy's (secret-free)
        inference_config into every record's provenance."""
        transitions = _read_jsonl(demo_run / "transitions.jsonl")
        assert transitions
        for rec in transitions:
            inf_cfg = rec["provenance"]["inference_config"]
            assert inf_cfg["model"] == "fake-llm"
            assert inf_cfg["api_key_env"] == _KEY_ENV
            # Only the env var NAME — never key material.
            assert _FAKE_KEY not in json.dumps(inf_cfg)

    def test_no_quality_flag_skips_chain(self, tmp_path: Path) -> None:
        mod = _load_demo_module("run_llm_demo_noquality")
        out = tmp_path / "no_quality"
        rc = mod.main(
            ["--mode", "fake", "--seeds", "42", "--ticks", "2",
             "--no-quality", "--output", str(out)]
        )
        assert rc == 0
        for name in _QUALITY_FILES:
            assert not (out / name).exists(), f"{name} written despite --no-quality"
        assert not (out / "dataset").exists()
        summary = json.loads((out / "summary.json").read_text("utf-8"))
        assert "quality" not in summary
