"""Transition recorder (K-06).

Append-only recorder that writes :class:`TransitionRecord` objects to
disk atomically. Failed writes go to quarantine; the main loop never
blocks on recorder failure.

Design rules (per main plan §4.6 and v1 SYSTEM_CONTRACT):
- Append-only. Existing records MUST NOT be modified or deleted.
- Atomic write per record: tmp file + rename. Crash leaves either the
  old state or the new state, never a partial write.
- Validation runs BEFORE append. Invalid records go to
  ``_quarantine/`` with a sidecar ``.reason.json``; they never enter
  the published dataset.
- Recorder failure MUST NOT crash the world. The kernel logs and
  continues; the manifest records the gap.
- Manifest is a single JSON file per run with: record_count,
  schema_version, producer_id, producer_version, first_tick, last_tick,
  hashes, quarantine_count, created_at, closed_at.

Provenance: extracted from ``current/worldloop/core/runtime/
transition_recorder.py`` (v1.0.0 tag); re-implemented in kernel with
the same atomic-write / quarantine / manifest discipline.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from worldloop_kernel.transition import TransitionRecord, PROTOCOL_SCHEMA_VERSION
from worldloop_kernel.validation import (
    ValidationReport,
    ValidationError,
    validate_transition,
)

__all__ = [
    "RecorderManifest",
    "RecorderError",
    "TransitionRecorder",
]


class RecorderError(RuntimeError):
    """Raised when the recorder encounters an internal error.

    The recorder catches I/O errors internally and routes the record to
    quarantine; this exception is reserved for programmer errors (e.g.,
    appending after close).
    """


@dataclass(frozen=True)
class RecorderManifest:
    """Per-run manifest summarizing the recorder's state.

    Attributes
    ----------
    record_count:
        Number of records successfully written to the published dataset.
    quarantine_count:
        Number of records routed to quarantine (validation failure or
        write error).
    schema_version:
        Kernel protocol schema version (see
        :data:`PROTOCOL_SCHEMA_VERSION`).
    producer_id:
        Stable identifier of the world implementation.
    producer_version:
        Version of the world implementation.
    first_tick:
        Tick of the first successfully appended record, or ``None`` if
        no records were appended.
    last_tick:
        Tick of the last successfully appended record, or ``None`` if
        no records were appended.
    state_before_hashes:
        Tuple of ``state_before_hash`` values, one per appended record.
        Useful for chain integrity verification.
    state_after_hashes:
        Tuple of ``state_after_hash`` values, one per appended record.
    created_at:
        Unix timestamp (seconds) when the recorder was created.
    closed_at:
        Unix timestamp (seconds) when the recorder was closed, or
        ``None`` if still open.
    output_dir:
        Absolute path to the published dataset directory.
    quarantine_dir:
        Absolute path to the quarantine directory.
    """

    record_count: int
    quarantine_count: int
    schema_version: str
    producer_id: str
    producer_version: str
    first_tick: int | None
    last_tick: int | None
    state_before_hashes: tuple[str, ...]
    state_after_hashes: tuple[str, ...]
    created_at: float
    closed_at: float | None
    output_dir: str
    quarantine_dir: str


# ---------------------------------------------------------------------------
# TransitionRecorder
# ---------------------------------------------------------------------------


# Filename template for published records. Tick is zero-padded to 10
# digits to preserve lexical sort order on most filesystems.
_RECORD_FILENAME_TEMPLATE = "t{tick:010d}.json"

# Manifest filename (single file per run, written on close).
_MANIFEST_FILENAME = "manifest.json"

# Quarantine subdirectory name (underscore prefix keeps it out of the
# published dataset glob).
_QUARANTINE_DIRNAME = "_quarantine"


class TransitionRecorder:
    """Append-only recorder for :class:`TransitionRecord` objects.

    The recorder writes one JSON file per record to ``output_dir``.
    Invalid records (failed validation) and records that fail to write
    are routed to ``output_dir/_quarantine/`` with a sidecar
    ``.reason.json`` describing the failure.

    The recorder is context-manager-friendly::

        with TransitionRecorder(Path("runs/x"), "world-id") as rec:
            rec.append(record)

    Or use ``close()`` explicitly. ``close()`` writes the manifest and
    marks the recorder as closed; further ``append()`` calls raise
    :class:`RecorderError`.
    """

    def __init__(
        self,
        output_dir: Path,
        world_id: str,
        *,
        producer_version: str = "0.0.0",
        validate: bool = True,
    ) -> None:
        """Initialize the recorder.

        Args:
            output_dir: Directory where the published dataset is written.
                Created if it does not exist.
            world_id: Stable identifier of the world implementation.
                Written into the manifest.
            producer_version: Version of the world implementation.
                Written into the manifest.
            validate: If True (default), run :func:`validate_transition`
                on each record before writing. Invalid records go to
                quarantine. If False, skip validation (use with care —
                only for trusted in-process recorders).
        """
        self._output_dir = Path(output_dir).resolve()
        self._quarantine_dir = self._output_dir / _QUARANTINE_DIRNAME
        self._world_id = world_id
        self._producer_version = producer_version
        self._validate = validate

        self._closed = False
        self._created_at = time.time()
        self._closed_at: float | None = None

        self._record_count = 0
        self._quarantine_count = 0
        self._first_tick: int | None = None
        self._last_tick: int | None = None
        self._state_before_hashes: list[str] = []
        self._state_after_hashes: list[str] = []

        # Track producer_id from the first record (assumed stable for
        # the run). Defaults to world_id.
        self._producer_id: str = world_id

        # Create directories.
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._quarantine_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "TransitionRecorder":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Always close, even on exception. The manifest reflects what
        # was successfully written before the exception.
        self.close()
        # Do not suppress the exception.
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(self, record: TransitionRecord) -> None:
        """Append one record to the dataset.

        If validation is enabled and the record fails, OR if the write
        fails, the record is routed to quarantine with a sidecar
        ``.reason.json``. The recorder does NOT raise on validation or
        write failure — the main loop continues.

        Args:
            record: The :class:`TransitionRecord` to append.

        Raises:
            RecorderError: If the recorder has been closed.
        """
        if self._closed:
            raise RecorderError(
                "cannot append to a closed recorder; create a new recorder "
                "for a new run"
            )

        # Capture producer_id from the first record (defensive — should
        # match world_id passed at construction).
        if self._record_count == 0 and self._quarantine_count == 0:
            self._producer_id = record.producer_id

        # Step 1: validate (if enabled).
        report: ValidationReport | None = None
        if self._validate:
            try:
                report = validate_transition(record)
            except ValidationError as exc:
                # Malformed record — quarantine immediately.
                self._quarantine_record(record, reason="malformed", error=str(exc))
                return
            if not report.passed:
                self._quarantine_record(
                    record,
                    reason="validation_failed",
                    report=self._serialize_report(report),
                )
                return

        # Step 2: atomic write to output_dir.
        try:
            self._atomic_write(record)
        except OSError as exc:
            # Write failed — route to quarantine with the OS error.
            self._quarantine_record(
                record,
                reason="write_failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            return

        # Step 3: update manifest state.
        self._record_count += 1
        if self._first_tick is None:
            self._first_tick = record.tick
        self._last_tick = record.tick
        self._state_before_hashes.append(record.state_before_hash)
        self._state_after_hashes.append(record.state_after_hash)

    def flush(self) -> None:
        """Flush any buffered state to disk.

        The recorder writes each record atomically on ``append()`` and
        does not buffer; ``flush()`` is therefore a no-op. It exists for
        API symmetry with file-like objects and to provide a hook for
        future buffering.
        """
        # No buffer to flush — atomic write per record.
        return None

    def close(self) -> None:
        """Close the recorder and write the manifest.

        After ``close()``, further ``append()`` calls raise
        :class:`RecorderError`. ``close()`` is idempotent.
        """
        if self._closed:
            return
        self._closed = True
        self._closed_at = time.time()
        self._write_manifest()

    def manifest(self) -> RecorderManifest:
        """Return a snapshot of the recorder's current manifest state.

        The manifest is also written to ``output_dir/manifest.json`` on
        :meth:`close`. This method returns the in-memory state, useful
        for live monitoring.
        """
        return RecorderManifest(
            record_count=self._record_count,
            quarantine_count=self._quarantine_count,
            schema_version=PROTOCOL_SCHEMA_VERSION,
            producer_id=self._producer_id,
            producer_version=self._producer_version,
            first_tick=self._first_tick,
            last_tick=self._last_tick,
            state_before_hashes=tuple(self._state_before_hashes),
            state_after_hashes=tuple(self._state_after_hashes),
            created_at=self._created_at,
            closed_at=self._closed_at,
            output_dir=str(self._output_dir),
            quarantine_dir=str(self._quarantine_dir),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _atomic_write(self, record: TransitionRecord) -> None:
        """Atomically write one record to ``output_dir``.

        Uses tmp file + os.replace for atomicity. The tmp file is
        created in the same directory to ensure ``os.replace`` is atomic
        on the same filesystem.
        """
        filename = _RECORD_FILENAME_TEMPLATE.format(tick=record.tick)
        target_path = self._output_dir / filename
        # Serialize once (we need the bytes for both tmp write and
        # checksum, if we ever add one).
        payload = self._serialize_record(record)
        # tempfile.NamedTemporaryFile + os.replace gives atomic rename.
        # Use delete=False so we can rename it ourselves.
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=self._output_dir,
            prefix=f".{filename}.",
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(payload)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        # os.replace is atomic on POSIX and Windows for same-filesystem
        # renames. If the target exists (rare — same tick appended
        # twice), it is overwritten.
        os.replace(tmp_path, target_path)

    def _quarantine_record(
        self,
        record: TransitionRecord,
        *,
        reason: str,
        error: str | None = None,
        report: Mapping[str, Any] | None = None,
    ) -> None:
        """Route a record to quarantine with a sidecar ``.reason.json``.

        The record is written to ``quarantine_dir/t{tick:010d}.json``
        and the reason is written to
        ``quarantine_dir/t{tick:010d}.reason.json``. If a record with
        the same tick is already in quarantine (rare — duplicate tick),
        a counter suffix is appended to avoid overwrite.
        """
        self._quarantine_count += 1
        base = _RECORD_FILENAME_TEMPLATE.format(tick=record.tick)
        record_path = self._quarantine_dir / base
        reason_path = self._quarantine_dir / base.replace(".json", ".reason.json")
        # Avoid overwriting an existing quarantined record with the same
        # tick. Append a counter.
        counter = 1
        while record_path.exists():
            record_path = self._quarantine_dir / base.replace(
                ".json", f".{counter}.json"
            )
            reason_path = self._quarantine_dir / base.replace(
                ".json", f".{counter}.reason.json"
            )
            counter += 1

        # Write record (best-effort — if this fails too, we lose the
        # record but the main loop continues).
        try:
            payload = self._serialize_record(record)
            with open(record_path, "w", encoding="utf-8") as f:
                f.write(payload)
        except OSError:
            # We tried. Drop the record; the quarantine_count still
            # reflects the attempt.
            return

        # Write reason sidecar.
        reason_payload: dict[str, Any] = {
            "reason": reason,
            "tick": record.tick,
            "producer_id": record.producer_id,
            "state_before_hash": record.state_before_hash,
            "state_after_hash": record.state_after_hash,
            "quarantined_at": time.time(),
        }
        if error is not None:
            reason_payload["error"] = error
        if report is not None:
            reason_payload["validation_report"] = dict(report)
        try:
            with open(reason_path, "w", encoding="utf-8") as f:
                json.dump(reason_payload, f, indent=2, sort_keys=True)
        except OSError:
            # Best-effort; the record is already in quarantine.
            return

    def _write_manifest(self) -> None:
        """Write the manifest JSON to ``output_dir/manifest.json``."""
        manifest = self.manifest()
        # Use dataclasses.asdict for clean JSON. Tuples become lists,
        # which is the standard JSON representation.
        from dataclasses import asdict

        payload = asdict(manifest)
        manifest_path = self._output_dir / _MANIFEST_FILENAME
        # Atomic write: tmp + rename.
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=self._output_dir,
            prefix=".manifest.",
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            json.dump(payload, tmp, indent=2, sort_keys=True)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, manifest_path)

    @staticmethod
    def _serialize_record(record: TransitionRecord) -> str:
        """Serialize a :class:`TransitionRecord` to a JSON string.

        The kernel does NOT commit to a stable on-disk JSON schema in
        K-06 — the schema is finalized in K-09 (wheel build) along with
        a backwards-compatibility policy. For now, we use
        :func:`dataclasses.asdict` + :mod:`json` with sorted keys.

        Note: ``Mapping`` fields (candidate_actions, executed_actions,
        receipts, provenance, missing_mask, payload, ...) are converted
        to plain dicts. Tuple fields become lists. Frozen dataclasses
        are converted recursively.
        """
        from dataclasses import asdict

        return json.dumps(asdict(record), indent=2, sort_keys=True, default=str)

    @staticmethod
    def _serialize_report(report: ValidationReport) -> dict[str, Any]:
        """Serialize a :class:`ValidationReport` to a JSON-safe dict."""
        from dataclasses import asdict

        return asdict(report)
