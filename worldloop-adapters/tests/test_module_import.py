"""F-07 audit fix: test entry point prints + verifies module import path.

Audit F-07 found that running pytest directly in a package directory could
silently import an old installed wheel from ``site-packages`` instead of
the workspace ``src/`` tree, producing "false green" test runs.

This test file is the F-07 gate for ``worldloop-adapters``: it
(1) prints ``worldloop_adapters.__file__`` to stdout (captured by pytest
    logs and visible in CI output for clean-room reproduction), and
(2) asserts the resolved path is inside the workspace ``src/`` tree,
    not inside ``site-packages``.
"""
from __future__ import annotations

from pathlib import Path


def test_module_import_path_in_workspace_src(capsys) -> None:
    """F-07: ``worldloop_adapters`` must be imported from workspace src/."""
    import worldloop_adapters

    assert hasattr(worldloop_adapters, "__file__"), "worldloop_adapters has no __file__"
    assert worldloop_adapters.__file__ is not None, "worldloop_adapters.__file__ is None"

    print(f"worldloop_adapters.__file__ = {worldloop_adapters.__file__}")
    print(f"worldloop_adapters.__version__ = {getattr(worldloop_adapters, '__version__', '<not set>')}")

    captured = capsys.readouterr()
    assert "worldloop_adapters.__file__" in captured.out

    path = Path(worldloop_adapters.__file__).resolve()
    assert "current" in path.parts, f"unexpected import path (not in workspace): {path}"
    assert "worldloop-adapters" in path.parts, f"unexpected import path: {path}"
    assert "src" in path.parts, f"unexpected import path (not in src/): {path}"
    assert "site-packages" not in path.parts, (
        f"F-07 violation: worldloop_adapters imported from site-packages: {path}"
    )
