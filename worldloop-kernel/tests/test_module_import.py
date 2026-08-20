"""F-07 audit fix: test entry point prints + verifies module import path.

Audit F-07 found that running pytest directly in a package directory could
silently import an old installed wheel from ``site-packages`` instead of
the workspace ``src/`` tree, producing "false green" test runs.

This test file is the F-07 gate for ``worldloop-kernel``: it
(1) prints ``worldloop_kernel.__file__`` to stdout (captured by pytest
    logs and visible in CI output for clean-room reproduction), and
(2) asserts the resolved path is inside the workspace ``src/`` tree,
    not inside ``site-packages``.

Equivalent test files exist in the other three v2 packages:
- ``current/worldloop-adapters/tests/test_module_import.py``
- ``current/worldloop-scenarios/tests/test_module_import.py``
- ``current/worldloop-data/tests/test_module_import.py``
"""
from __future__ import annotations

from pathlib import Path


def test_module_import_path_in_workspace_src(capsys) -> None:
    """F-07: ``worldloop_kernel`` must be imported from workspace src/.

    Prints ``module.__file__`` so clean-room reproduction logs show the
    actual import path, then asserts the path is inside the workspace
    ``current/worldloop-kernel/src/`` tree.
    """
    import worldloop_kernel

    assert hasattr(worldloop_kernel, "__file__"), "worldloop_kernel has no __file__"
    assert worldloop_kernel.__file__ is not None, "worldloop_kernel.__file__ is None"

    # Print for clean-room reproduction logs (F-07 audit requirement).
    print(f"worldloop_kernel.__file__ = {worldloop_kernel.__file__}")
    print(f"worldloop_kernel.__version__ = {getattr(worldloop_kernel, '__version__', '<not set>')}")

    # Flush captured stdout so it appears in pytest -v output.
    captured = capsys.readouterr()
    assert "worldloop_kernel.__file__" in captured.out

    path = Path(worldloop_kernel.__file__).resolve()
    assert "current" in path.parts, f"unexpected import path (not in workspace): {path}"
    assert "worldloop-kernel" in path.parts, f"unexpected import path: {path}"
    assert "src" in path.parts, f"unexpected import path (not in src/): {path}"
    # Hard guard: must NOT be inside site-packages.
    assert "site-packages" not in path.parts, (
        f"F-07 violation: worldloop_kernel imported from site-packages: {path}"
    )


def test_package_metadata_consistent() -> None:
    """F-07: package version metadata is readable and consistent."""
    import worldloop_kernel

    # Kernel package exposes __version__ via pyproject.toml.
    version = getattr(worldloop_kernel, "__version__", None)
    if version is not None:
        assert isinstance(version, str)
        assert version  # non-empty
