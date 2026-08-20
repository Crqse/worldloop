"""Canonical encoding and hashing (K-05).

Provides ``canonical_encode(value) -> bytes`` and ``hash_state(state) -> str``.
The kernel uses these to compute ``state_before_hash`` / ``state_after_hash``
in :class:`TransitionRecord` and to verify the round-trip invariant in
:func:`worldloop_kernel.diff_apply.apply_delta`.

Design rules (per ADR §3 and main plan §4.6):
- Key order MUST be deterministic (sorted lexicographically by encoded key
  for ``Mapping``).
- Float normalization: NaN/inf rejected; +/-0.0 unified to +0.0; rounding
  to 12 decimal places (about 1e-12 precision, well below typical
  simulation noise).
- ``None`` and missing fields MUST be encoded distinctly from empty
  collections (``None`` -> ``b"N"``; empty tuple -> ``b"()"``; empty
  mapping -> ``b"{}"``).
- Encoding MUST be stable across Python versions and platforms. We use
  only ASCII repr for scalars, UTF-8 for strings, and length-prefixed
  byte payloads to avoid ambiguity.
- ``bool`` is encoded distinctly from ``int`` (``True`` -> ``b"T"``,
  ``False`` -> ``b"F"``), even though ``isinstance(True, int)`` is ``True``
  in Python.

Provenance: extracted from ``current/worldloop/core/runtime/
transition_validation.py`` (v1.0.0 tag ``worldloop-v1.0.0-2026-07-26``);
the v1 codec is treated as a reference, not imported. The kernel
re-implements a minimal subset for v2 protocol.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import fields, is_dataclass
from typing import Any, Mapping

__all__ = [
    "CanonicalError",
    "canonical_encode",
    "hash_state",
    "HASH_PREFIX",
]


class CanonicalError(ValueError):
    """Raised when a value cannot be canonically encoded."""


# Hash algorithm prefix. Embedded in every ``state_before_hash`` /
# ``state_after_hash`` so consumers know which algorithm produced the
# digest. Bumping the algorithm (e.g., to blake2b) changes the prefix
# and is a backward-incompatible protocol change.
HASH_PREFIX = "sha256"

# Number of decimal places to round floats before encoding. 12 places
# gives ~1e-12 precision, well below typical simulation noise, and
# eliminates most cross-platform float representation differences
# (e.g., 0.1 + 0.2 vs 0.3).
_FLOAT_ROUND_PLACES = 12

# Single-byte type tags. Using distinct tags for every type avoids
# ambiguity when decoding (though the kernel currently only encodes,
# never decodes — these tags are for forward compatibility and for
# humans reading hex dumps).
_TAG_NONE = b"N"
_TAG_BOOL_TRUE = b"T"
_TAG_BOOL_FALSE = b"F"
_TAG_INT = b"i"
_TAG_FLOAT = b"f"
_TAG_STR = b"s"
_TAG_BYTES = b"b"
_TAG_TUPLE = b"("
_TAG_FROZENSET = b"#"
_TAG_MAPPING = b"M"
_TAG_DATACLASS = b"D"

# Distinct sentinels for empty collections. These MUST differ from
# ``_TAG_NONE`` so that ``None`` is never confused with an empty tuple
# or empty mapping.
_TAG_EMPTY_TUPLE = b"()"
_TAG_EMPTY_MAPPING = b"{}"


def _encode_str(value: str) -> bytes:
    """Length-prefixed UTF-8 encoding for strings."""
    encoded = value.encode("utf-8")
    return str(len(encoded)).encode("ascii") + b":" + encoded


def _encode_bytes(value: bytes) -> bytes:
    """Length-prefixed raw bytes encoding."""
    return str(len(value)).encode("ascii") + b":" + value


def _encode_float(value: float) -> bytes:
    """Encode a float with normalization.

    - NaN / inf are rejected (they break hash stability).
    - +0.0 and -0.0 are unified to +0.0.
    - The value is rounded to ``_FLOAT_ROUND_PLACES`` decimal places.
    """
    if math.isnan(value):
        raise CanonicalError(
            "NaN cannot be canonically encoded; it breaks hash stability"
        )
    if math.isinf(value):
        raise CanonicalError(
            "inf cannot be canonically encoded; it breaks hash stability"
        )
    # Unify -0.0 and +0.0. ``+0.0 == -0.0`` is True in Python, so we
    # add 0.0 to coerce both to +0.0.
    if value == 0.0:
        value = 0.0
    rounded = round(value, _FLOAT_ROUND_PLACES)
    # repr() gives the shortest round-trippable representation, which
    # is stable across Python versions for the same numeric value.
    return repr(rounded).encode("ascii")


def _encode_tuple(value: tuple) -> bytes:
    """Encode a tuple. Empty tuple gets a distinct sentinel."""
    if not value:
        return _TAG_EMPTY_TUPLE
    parts = [_TAG_TUPLE, str(len(value)).encode("ascii"), b":"]
    for item in value:
        parts.append(canonical_encode(item))
    return b"".join(parts)


def _encode_frozenset(value: frozenset) -> bytes:
    """Encode a frozenset by sorting items by their canonical encoding."""
    encoded_items = sorted(canonical_encode(item) for item in value)
    parts = [_TAG_FROZENSET, str(len(encoded_items)).encode("ascii"), b":"]
    parts.extend(encoded_items)
    return b"".join(parts)


def _encode_mapping(value: Mapping) -> bytes:
    """Encode a mapping with keys sorted by their canonical encoding."""
    if not value:
        return _TAG_EMPTY_MAPPING
    # Compute canonical encoding of each key, then sort by the encoded
    # bytes. This guarantees deterministic key order regardless of the
    # original insertion order.
    encoded_keys: list[tuple[bytes, Any]] = []
    for k in value.keys():
        k_enc = canonical_encode(k)
        encoded_keys.append((k_enc, k))
    encoded_keys.sort(key=lambda pair: pair[0])
    parts = [_TAG_MAPPING, str(len(encoded_keys)).encode("ascii"), b":"]
    for k_enc, k in encoded_keys:
        parts.append(k_enc)
        parts.append(canonical_encode(value[k]))
    return b"".join(parts)


def _encode_dataclass(value: Any) -> bytes:
    """Encode a frozen dataclass by iterating its declared fields in order.

    The class name is included so that two structurally identical
    dataclasses with different names produce different hashes (avoiding
    accidental collisions across types).
    """
    cls = type(value)
    cls_name = cls.__name__
    parts = [_TAG_DATACLASS, _encode_str(cls_name)]
    for f in fields(value):
        parts.append(_encode_str(f.name))
        parts.append(canonical_encode(getattr(value, f.name)))
    return b"".join(parts)


def canonical_encode(value: Any) -> bytes:
    """Encode any kernel-compatible value into deterministic bytes.

    Supported types:
    - ``None`` -> ``b"N"``
    - ``bool`` -> ``b"T"`` / ``b"F"``
    - ``int`` -> ``b"i" + repr``
    - ``float`` -> ``b"f" + normalized repr`` (NaN/inf rejected)
    - ``str`` -> ``b"s" + length-prefixed UTF-8``
    - ``bytes`` / ``bytearray`` -> ``b"b" + length-prefixed raw``
    - ``tuple`` -> ``b"(" + count + items`` (empty -> ``b"()"``)
    - ``frozenset`` -> ``b"#" + count + sorted items``
    - ``Mapping`` -> ``b"M" + count + (key, value)*`` (empty -> ``b"{}"``)
    - dataclass instance -> ``b"D" + class name + (field name, value)*``

    Unsupported types raise :class:`CanonicalError`.

    The encoding is deterministic: the same value always produces the
    same bytes on any Python implementation >= 3.10. Floats are rounded
    to 12 decimal places to suppress cross-platform representation
    differences.
    """
    # Order matters: bool MUST be checked before int (isinstance(True, int)
    # is True in Python). We use identity checks for the singleton bools.
    if value is None:
        return _TAG_NONE
    if value is True:
        return _TAG_BOOL_TRUE
    if value is False:
        return _TAG_BOOL_FALSE
    if isinstance(value, int):
        return _TAG_INT + repr(value).encode("ascii")
    if isinstance(value, float):
        return _TAG_FLOAT + _encode_float(value)
    if isinstance(value, str):
        return _TAG_STR + _encode_str(value)
    if isinstance(value, (bytes, bytearray)):
        return _TAG_BYTES + _encode_bytes(bytes(value))
    if isinstance(value, tuple):
        return _encode_tuple(value)
    if isinstance(value, frozenset):
        return _encode_frozenset(value)
    if isinstance(value, Mapping):
        return _encode_mapping(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _encode_dataclass(value)
    raise CanonicalError(
        f"Cannot canonically encode value of type {type(value).__name__!r}; "
        "supported types are None, bool, int, float, str, bytes, tuple, "
        "frozenset, Mapping, and dataclass instances."
    )


def hash_state(state: Any) -> str:
    """Compute the canonical hash of a value.

    Returns a string of the form ``"sha256:<hex_digest>"``. The prefix
    identifies the hash algorithm so consumers can verify or migrate.

    For a :class:`worldloop_kernel.state.StateView`, the hash includes
    all observable state: ``meta``, ``entities``, ``fields``,
    ``relations``, ``registries``, ``population``, ``events``,
    ``capabilities``, and ``missing_mask``. Two states with the same
    hash are observationally identical; two states with different hashes
    differ in at least one observable field.
    """
    encoded = canonical_encode(state)
    digest = hashlib.sha256(encoded).hexdigest()
    return f"{HASH_PREFIX}:{digest}"
