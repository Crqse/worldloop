"""K-03 scaffold smoke tests.

These tests verify that the scaffold is importable, versioned, and free
of v1 Native imports. They do NOT test behavior — behavior tests land
in K-04~K-09.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


def test_package_importable():
    """``import worldloop_kernel`` must succeed."""
    importlib.import_module("worldloop_kernel")
    assert "worldloop_kernel" in sys.modules


def test_version_exposed():
    """``worldloop_kernel.__version__`` must be a non-empty string."""
    import worldloop_kernel

    assert isinstance(worldloop_kernel.__version__, str)
    assert worldloop_kernel.__version__


def test_version_is_0_1_3():
    """Package version must match pyproject.toml (0.1.3)."""
    import worldloop_kernel

    assert worldloop_kernel.__version__ == "0.1.3"


@pytest.mark.parametrize(
    "module_name",
    [
        "capability",
        "state",
        "action",
        "transition",
        "protocol",
        "canonical",
        "diff_apply",
        "validation",
        "recorder",
        "replay",
        "engine",
    ],
)
def test_submodule_importable(module_name):
    """All 11 planned submodules must be importable as stubs."""
    importlib.import_module(f"worldloop_kernel.{module_name}")


def test_no_v1_native_imports():
    """The kernel MUST NOT import any v1 Native code.

    This is a hard ADR §3 constraint. We verify it by checking that
    no ``current.worldloop`` or ``core.L1_*`` / ``core.L2_*`` / ... /
    ``core.L5_*`` module appears in ``sys.modules`` after importing the
    kernel package and every planned submodule.
    """
    import worldloop_kernel  # noqa: F401
    for name in (
        "worldloop_kernel.capability",
        "worldloop_kernel.state",
        "worldloop_kernel.action",
        "worldloop_kernel.transition",
        "worldloop_kernel.protocol",
        "worldloop_kernel.canonical",
        "worldloop_kernel.diff_apply",
        "worldloop_kernel.validation",
        "worldloop_kernel.recorder",
        "worldloop_kernel.replay",
        "worldloop_kernel.engine",
    ):
        importlib.import_module(name)

    forbidden_prefixes = (
        "current.worldloop",
        "core.L1_",
        "core.L2_",
        "core.L3_",
        "core.L4_",
        "core.L5_",
        "core.bridge",
        "core.runtime",
    )
    leaked = [m for m in sys.modules if m.startswith(forbidden_prefixes)]
    assert not leaked, f"kernel imports v1 Native modules: {leaked}"


def test_no_third_party_runtime_deps():
    """The kernel core MUST NOT import torch / numpy / pettingzoo at runtime.

    ADR §3 hard constraint: core depends only on Python standard library.
    Optional dependencies (jsonschema) live in extras and are imported
    lazily by the consumer that needs them.
    """
    import worldloop_kernel  # noqa: F401
    for name in (
        "worldloop_kernel.capability",
        "worldloop_kernel.state",
        "worldloop_kernel.action",
        "worldloop_kernel.transition",
        "worldloop_kernel.protocol",
        "worldloop_kernel.canonical",
        "worldloop_kernel.diff_apply",
        "worldloop_kernel.validation",
        "worldloop_kernel.recorder",
        "worldloop_kernel.replay",
        "worldloop_kernel.engine",
    ):
        importlib.import_module(name)

    forbidden = ("torch", "numpy", "pettingzoo", "gymnasium", "gym")
    leaked = [m for m in forbidden if m in sys.modules]
    assert not leaked, f"kernel pulls in third-party runtime deps: {leaked}"


def test_pyproject_python_requires_ge_3_10():
    """K-03 scaffold MUST require Python >= 3.10 (per main plan §10.1)."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert 'requires-python = ">=3.10"' in text


def test_pyproject_zero_runtime_deps():
    """K-03 scaffold MUST have zero runtime dependencies (per ADR §3)."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    # The literal "dependencies = []" must appear in [project].
    assert "dependencies = []" in text
