"""M8 group-aware split utilities + leakage assertions.

Beta B5 / M8 discipline (per PROJECT_CONVENTIONS strict group split):

- Samples sharing the same *group key* MUST land in the same split.
  A group key bundles everything that makes two rows statistically
  dependent: same seed, same episode family, same branch group, and
  same counterfactual fork point (branch siblings share the parent's
  ``state_before_hash`` at the fork tick).
- Random ROW-level splitting is FORBIDDEN (main plan §14.6). This
  module does not even expose a row-level API; :func:`assert_no_group_leakage`
  is the guard that catches such splits if produced elsewhere.
- Scalers / feature selection must be fit on train only — that is the
  caller's job; this module only owns split assignment + verification.

Episode-id conventions consumed here (produced by
``scripts/generate_m8_dataset.py``):

- Main episodes:   ``seed{S}_run{i}``
- Branch episodes: ``seed{S}_run{i}_cf_t{tick}_b{j}``  (counterfactual
  siblings of the main episode at fork tick ``tick``)

``episode_family`` strips the ``_cf_...`` suffix, so a branch episode
always shares its family (and therefore its group) with its parent.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from worldloop_data.evaluation.data_loader import TransitionSample

__all__ = [
    "GroupKey",
    "GroupLeakageError",
    "episode_family",
    "group_key_for_sample",
    "grouped_split",
    "assert_no_group_leakage",
    "assert_branch_siblings_together",
]


#: Suffix pattern marking counterfactual branch episodes.
_CF_SUFFIX_RE = re.compile(r"_cf_t\d+_b\d+$")

#: The only split strategies this module accepts. Anything row-level
#: (e.g., "random_rows") is rejected by construction.
_ALLOWED_STRATEGIES = ("seed", "episode_family")


class GroupLeakageError(AssertionError):
    """Raised when dependent samples cross split boundaries."""


@dataclass(frozen=True)
class GroupKey:
    """Split-atomic group identity for one sample.

    Attributes
    ----------
    seed:
        RNG seed string from provenance. Primary grouping factor: two
        rows from the same seed are NEVER split apart.
    family:
        Episode family (parent episode id with any ``_cf_t*_b*`` suffix
        stripped). Branch siblings share the family of their parent.
    """

    seed: str
    family: str


def episode_family(episode_id: str) -> str:
    """Return the episode family (parent id) for ``episode_id``.

    Branch episodes ``<parent>_cf_t{tick}_b{j}`` map to ``<parent>``;
    plain episodes map to themselves.
    """
    return _CF_SUFFIX_RE.sub("", episode_id)


def is_branch_episode(episode_id: str) -> bool:
    """True iff ``episode_id`` is a counterfactual branch episode."""
    return _CF_SUFFIX_RE.search(episode_id) is not None


def group_key_for_sample(sample: TransitionSample) -> GroupKey:
    """Build the split-atomic :class:`GroupKey` for a sample."""
    return GroupKey(
        seed=str(sample.seed),
        family=episode_family(sample.episode_id),
    )


def grouped_split(
    samples: Iterable[TransitionSample],
    seed_split_map: Mapping[str, str],
    *,
    strategy: str = "seed",
) -> dict[str, list[TransitionSample]]:
    """Assign samples to splits by GROUP (seed), never by row.

    Parameters
    ----------
    samples:
        Samples to assign. Branch siblings are automatically kept with
        their parent because assignment is a pure function of the
        group key (seed), never of the row index.
    seed_split_map:
        Explicit pre-registered ``{seed_str: split_name}`` map — the
        single source of truth, mirroring ``ExporterConfig.seed_split_map``.
    strategy:
        Must be one of ``"seed"`` / ``"episode_family"``. Any other
        value (in particular anything row-level like ``"random_rows"``)
        raises ``ValueError`` — random row splitting is forbidden by
        design (main plan §14.6).

    Returns
    -------
    dict[str, list[TransitionSample]]
        ``split_name -> samples``. Samples whose seed is missing from
        the map raise ``GroupLeakageError`` (silent drop would hide
        budget mistakes).
    """
    if strategy not in _ALLOWED_STRATEGIES:
        raise ValueError(
            f"forbidden split strategy {strategy!r}: random/row-level "
            f"splitting is not allowed; use one of {_ALLOWED_STRATEGIES}"
        )
    out: dict[str, list[TransitionSample]] = {}
    for s in samples:
        key = group_key_for_sample(s)
        split = seed_split_map.get(key.seed)
        if split is None:
            raise GroupLeakageError(
                f"sample from seed {key.seed!r} (episode {s.episode_id!r}) "
                f"has no entry in seed_split_map — refusing to guess"
            )
        out.setdefault(split, []).append(s)
    return out


def assert_no_group_leakage(
    splits: Mapping[str, Sequence[TransitionSample]],
) -> None:
    """Assert that no group key spans more than one split.

    Checks three progressively finer keys:

    1. ``seed`` — same seed in two splits (also catches random row
       splits, whose rows scatter seeds across splits).
    2. ``episode_family`` — same episode family (parent + its branch
       episodes) in two splits.
    3. branch fork point — ``(seed, tick, state_before_hash)`` of any
       BRANCH sample must not appear (as a fork point of a branch or
       factual sibling) in another split. This is the "counterfactual
       siblings from the same起点 never cross splits" guard.

    Raises
    ------
    GroupLeakageError
        With a description of the first violations found.
    """
    seed_to_splits: dict[str, set[str]] = {}
    family_to_splits: dict[str, set[str]] = {}
    fork_to_splits: dict[tuple[str, int, str], set[str]] = {}

    for split_name, split_samples in splits.items():
        for s in split_samples:
            key = group_key_for_sample(s)
            seed_to_splits.setdefault(key.seed, set()).add(split_name)
            family_to_splits.setdefault(key.family, set()).add(split_name)
            # Fork-point key: every sample is a potential sibling of a
            # branch at (seed, tick, state_before_hash). Branch records
            # share the parent's state_before_hash at the fork tick.
            fork_key = (key.seed, s.tick, s.state_before_hash)
            fork_to_splits.setdefault(fork_key, set()).add(split_name)

    violations: list[str] = []
    for seed, names in sorted(seed_to_splits.items()):
        if len(names) > 1:
            violations.append(
                f"seed {seed!r} appears in splits {sorted(names)}"
            )
    for family, names in sorted(family_to_splits.items()):
        if len(names) > 1:
            violations.append(
                f"episode family {family!r} appears in splits {sorted(names)}"
            )
    for fork_key, names in sorted(fork_to_splits.items()):
        if len(names) > 1:
            violations.append(
                f"fork point (seed={fork_key[0]}, tick={fork_key[1]}, "
                f"state_before_hash={fork_key[2][:16]}...) appears in "
                f"splits {sorted(names)}"
            )
    if violations:
        raise GroupLeakageError(
            "group leakage detected across splits:\n  - "
            + "\n  - ".join(violations[:20])
        )


def assert_branch_siblings_together(
    splits: Mapping[str, Sequence[TransitionSample]],
) -> None:
    """Assert every branch episode lives in the same split as its parent.

    Complements :func:`assert_no_group_leakage` with an explicit
    parent↔branch check (usable even when the parent factual sample was
    filtered out — the family itself must stay split-pure).
    """
    family_to_splits: dict[str, set[str]] = {}
    for split_name, split_samples in splits.items():
        for s in split_samples:
            family_to_splits.setdefault(
                episode_family(s.episode_id), set()
            ).add(split_name)
    bad = {
        fam: sorted(names)
        for fam, names in family_to_splits.items()
        if len(names) > 1
    }
    if bad:
        raise GroupLeakageError(
            f"branch sibling families cross splits: {bad}"
        )
