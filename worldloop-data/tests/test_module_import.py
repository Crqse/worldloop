"""F-07 audit fix: test entry point prints + verifies module import path.

Audit F-07 found that running pytest directly in a package directory could
silently import an old installed wheel from ``site-packages`` instead of
the workspace ``src/`` tree, producing "false green" test runs.

This test file is the F-07 gate for ``worldloop-data``: it
(1) prints ``worldloop_data.__file__`` to stdout (captured by pytest
    logs and visible in CI output for clean-room reproduction), and
(2) asserts the resolved path is inside the workspace ``src/`` tree,
    not inside ``site-packages``.

A separate ``TestModuleImportPath`` class in
``tests/test_state_materializer.py`` covers the
``worldloop_data.evaluation.state_materializer`` submodule; this file
covers the package root and the ``evaluation`` subpackage.
"""
from __future__ import annotations

from pathlib import Path


def _assert_src_layout(path: Path, package_dir: str, package_root: str) -> None:
    """Assert ``path`` lies under ``<package_root>/src/<package_dir>/``.

    Works for both the mother repo layout (``current/<package_root>/src/…``)
    and the independent public release tree (``<package_root>/src/…``).
    Still hard-fails on site-packages imports — the real guard F-07 was
    written to catch.
    """
    assert path.parent.name == package_dir, f"unexpected package dir: {path}"
    assert path.parent.parent.name == "src", f"unexpected import path (not under src/): {path}"
    assert path.parent.parent.parent.name == package_root, f"unexpected package-root dir: {path}"
    assert package_root in path.parts, f"unexpected import path: {path}"
    assert "src" in path.parts, f"unexpected import path (not in src/): {path}"
    assert "site-packages" not in path.parts, (
        f"F-07 violation: module imported from site-packages instead of src/ tree: {path}"
    )


def test_package_root_import_path_in_workspace_src(capsys) -> None:
    """F-07: ``worldloop_data`` package root must be imported from workspace src/."""
    import worldloop_data

    assert hasattr(worldloop_data, "__file__"), "worldloop_data has no __file__"
    assert worldloop_data.__file__ is not None, "worldloop_data.__file__ is None"

    print(f"worldloop_data.__file__ = {worldloop_data.__file__}")
    print(f"worldloop_data.__version__ = {getattr(worldloop_data, '__version__', '<not set>')}")

    captured = capsys.readouterr()
    assert "worldloop_data.__file__" in captured.out

    path = Path(worldloop_data.__file__).resolve()
    _assert_src_layout(path, package_dir="worldloop_data", package_root="worldloop-data")


def test_evaluation_subpackage_import_path_in_workspace_src(capsys) -> None:
    """F-07: ``worldloop_data.evaluation`` subpackage must be from workspace src/."""
    import worldloop_data.evaluation as evaluation

    assert hasattr(evaluation, "__file__"), "evaluation has no __file__"
    assert evaluation.__file__ is not None, "evaluation.__file__ is None"

    print(f"worldloop_data.evaluation.__file__ = {evaluation.__file__}")

    captured = capsys.readouterr()
    assert "worldloop_data.evaluation.__file__" in captured.out

    # evaluation lives one level deeper under worldloop_data; still enforce
    # <package-root>/src/<package_dir>/evaluation/__init__.py
    path = Path(evaluation.__file__).resolve()
    assert path.parent.name == "evaluation", f"unexpected subpackage dir: {path}"
    assert path.parent.parent.name == "worldloop_data", f"unexpected package dir: {path}"
    assert path.parent.parent.parent.name == "src", f"unexpected import path (not under src/): {path}"
    assert path.parent.parent.parent.parent.name == "worldloop-data", f"unexpected package-root dir: {path}"
    assert "worldloop-data" in path.parts, f"unexpected import path: {path}"
    assert "src" in path.parts, f"unexpected import path (not in src/): {path}"
    assert "site-packages" not in path.parts, (
        f"F-07 violation: worldloop_data.evaluation imported from site-packages: {path}"
    )


def test_data_loader_import_path_in_workspace_src(capsys) -> None:
    """F-07: ``DataLoader`` must be imported from workspace src/."""
    from worldloop_data.evaluation import data_loader

    assert hasattr(data_loader, "__file__"), "data_loader has no __file__"
    assert data_loader.__file__ is not None

    print(f"worldloop_data.evaluation.data_loader.__file__ = {data_loader.__file__}")

    captured = capsys.readouterr()
    assert "data_loader.__file__" in captured.out

    # Module-level file: <root>/src/worldloop_data/evaluation/data_loader.py
    # -> parent = evaluation, grandparent = worldloop_data, g-gparent = src
    path = Path(data_loader.__file__).resolve()
    assert path.parent.name == "evaluation", f"unexpected subpackage dir: {path}"
    assert path.parent.parent.name == "worldloop_data", f"unexpected package dir: {path}"
    assert path.parent.parent.parent.name == "src", f"unexpected import path (not under src/): {path}"
    assert path.parent.parent.parent.parent.name == "worldloop-data", f"unexpected package-root dir: {path}"
    assert "worldloop-data" in path.parts, f"unexpected import path: {path}"
    assert "src" in path.parts, f"unexpected import path (not in src/): {path}"
    assert "site-packages" not in path.parts, (
        f"F-07 violation: data_loader imported from site-packages: {path}"
    )


def test_baselines_import_path_in_workspace_src(capsys) -> None:
    """F-07: ``baselines`` module must be imported from workspace src/."""
    from worldloop_data.evaluation import baselines

    assert hasattr(baselines, "__file__"), "baselines has no __file__"
    assert baselines.__file__ is not None

    print(f"worldloop_data.evaluation.baselines.__file__ = {baselines.__file__}")

    captured = capsys.readouterr()
    assert "baselines.__file__" in captured.out

    path = Path(baselines.__file__).resolve()
    assert path.parent.name == "evaluation", f"unexpected subpackage dir: {path}"
    assert path.parent.parent.name == "worldloop_data", f"unexpected package dir: {path}"
    assert path.parent.parent.parent.name == "src", f"unexpected import path (not under src/): {path}"
    assert path.parent.parent.parent.parent.name == "worldloop-data", f"unexpected package-root dir: {path}"
    assert "worldloop-data" in path.parts, f"unexpected import path: {path}"
    assert "src" in path.parts, f"unexpected import path (not in src/): {path}"
    assert "site-packages" not in path.parts, (
        f"F-07 violation: baselines imported from site-packages: {path}"
    )
