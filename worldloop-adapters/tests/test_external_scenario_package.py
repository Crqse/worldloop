"""Tests for ExternalScenarioPackage (B4) — factory-generic, no env required.

These tests exercise the package dataclass and its pipeline-protocol
shape WITHOUT PettingZoo: the world factory is an arbitrary stub, proving
the package is not hard-wired to Simple Spread (or any specific env).
"""
from __future__ import annotations

import dataclasses

import pytest

from worldloop_adapters.scenario_package import (
    ExternalScenarioPackage,
    ExternalScenarioRef,
    ExternalSpecView,
    hash_world_parameters,
)


class _StubWorld:
    """Minimal stand-in world; only identity matters for these tests."""

    def __init__(self, seed: int) -> None:
        self.seed = seed


def _make_package(**overrides):
    kwargs = dict(
        scenario_id="external-stub-scenario",
        world_factory=lambda seed: _StubWorld(seed),
        world_parameters_hash=hash_world_parameters({"kind": "stub", "n": 2}),
        metadata={"source": "stub", "world_parameters": {"kind": "stub", "n": 2}},
    )
    kwargs.update(overrides)
    return ExternalScenarioPackage(**kwargs)


# ---------------------------------------------------------------------------
# Construction + validation
# ---------------------------------------------------------------------------


class TestExternalScenarioPackageConstruction:
    def test_constructs_with_arbitrary_factory(self):
        """The package is factory-generic — any callable works."""
        pkg = _make_package()
        world = pkg.world_factory(7)
        assert isinstance(world, _StubWorld)
        assert world.seed == 7

    def test_is_frozen(self):
        pkg = _make_package()
        with pytest.raises(dataclasses.FrozenInstanceError):
            pkg.scenario_id = "mutated"  # type: ignore[misc]

    def test_empty_scenario_id_rejected(self):
        with pytest.raises(ValueError, match="scenario_id"):
            _make_package(scenario_id="")

    def test_empty_hash_rejected(self):
        with pytest.raises(ValueError, match="world_parameters_hash"):
            _make_package(world_parameters_hash="")

    def test_non_callable_factory_rejected(self):
        with pytest.raises(ValueError, match="world_factory"):
            _make_package(world_factory="not-callable")

    def test_metadata_defaults_to_empty(self):
        pkg = ExternalScenarioPackage(
            scenario_id="s",
            world_factory=lambda seed: _StubWorld(seed),
            world_parameters_hash="sha256:abc",
        )
        assert dict(pkg.metadata) == {}


# ---------------------------------------------------------------------------
# Pipeline-protocol shape (duck-typed against run_pipeline/exporter usage)
# ---------------------------------------------------------------------------


class TestPipelineProtocolShape:
    def test_spec_scenario_scenario_id(self):
        """run_pipeline reads ``package.spec.scenario.scenario_id``."""
        pkg = _make_package()
        assert pkg.spec.scenario.scenario_id == "external-stub-scenario"

    def test_spec_to_dict(self):
        """PlainDatasetExporter calls ``spec.to_dict()`` for world_parameters/."""
        pkg = _make_package()
        d = pkg.spec.to_dict()
        assert d["scenario"]["scenario_id"] == "external-stub-scenario"
        assert d["world_parameters"] == {"kind": "stub", "n": 2}
        assert d["source"] == "external"

    def test_world_parameters_hash_attribute(self):
        pkg = _make_package()
        assert pkg.world_parameters_hash.startswith("sha256:")

    def test_spec_view_types(self):
        pkg = _make_package()
        assert isinstance(pkg.spec, ExternalSpecView)
        assert isinstance(pkg.spec.scenario, ExternalScenarioRef)


# ---------------------------------------------------------------------------
# hash_world_parameters
# ---------------------------------------------------------------------------


class TestHashWorldParameters:
    def test_deterministic(self):
        params = {"a": 1, "b": 2.5}
        assert hash_world_parameters(params) == hash_world_parameters(params)

    def test_key_order_insensitive(self):
        assert hash_world_parameters({"a": 1, "b": 2}) == hash_world_parameters(
            {"b": 2, "a": 1}
        )

    def test_parameter_change_changes_hash(self):
        assert hash_world_parameters({"n": 2}) != hash_world_parameters({"n": 3})
