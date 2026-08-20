"""Release hygiene checks for the installable data package."""

from pathlib import Path


def test_runtime_outputs_never_live_inside_package_root():
    package_root = Path(__file__).resolve().parents[1]
    assert not (package_root / "runs").exists(), (
        "runtime outputs must be written to the workspace-level runs/ "
        "directory, never current/worldloop-data/runs/"
    )
