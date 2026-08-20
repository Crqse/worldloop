"""M8 treatment construction (A/B/C/D/D-matched) + paired comparison stats.

Treatments (same generation budget: same scenario, same seeds, same
ticks; only the composition of TRAINING data varies — the held-out
evaluation set is IDENTICAL across treatments):

    A_small          Small baseline: conventional transitions restricted
                     to a deterministic tick-prefix of each training
                     episode (small_fraction of ticks). NOT a random row
                     subsample — prefix keeps the temporal structure.
    B_random         Random-policy-only conventional trajectories
                     (produced by a separate RandomPolicy-only run over
                     the same train seeds / tick budget).
    C_transition     Multi-policy conventional transitions (no branch
                     records). The reference treatment.
    D_counterfactual C + counterfactual branch transitions from the
                     same training seeds (KernelBranchScheduler forks).
    D_matched        D deterministically downsampled to |C| samples —
                     controls for "D is just more rows".

Same-budget comparison rule: all treatments are trained with the SAME
model class, hyperparameters, and evaluation protocol; only the
training rows differ. D_matched vs C is the PRIMARY comparison.

Statistics: model-seed repeats are first aggregated within each
independent data seed.  Paired bootstrap inference is then performed
over data-seed means.  Model seeds describe training stability; they do
not increase the environmental sample size.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from worldloop_data.evaluation.data_loader import TransitionSample
from worldloop_data.evaluation.grouped_split import is_branch_episode

__all__ = [
    "TREATMENT_NAMES",
    "is_counterfactual_sample",
    "prefix_subset",
    "downsample_matched",
    "build_treatments",
    "PairedStats",
    "paired_stats",
    "aggregate_model_seed_diffs",
    "verdict_from_stats",
]

TREATMENT_NAMES = (
    "A_small",
    "B_random",
    "C_transition",
    "D_counterfactual",
    "D_matched",
)

#: Absolute mean-diff threshold under which a CI-spanning-zero result is
#: called NULL rather than INCONCLUSIVE (pre-registered in M8 prereg).
NULL_EPSILON = 0.005


def is_counterfactual_sample(sample: TransitionSample) -> bool:
    """True iff the sample is a counterfactual branch transition."""
    if sample.policy_id == "counterfactual_branch":
        return True
    return is_branch_episode(sample.episode_id)


def prefix_subset(
    samples: Sequence[TransitionSample],
    fraction: float,
) -> list[TransitionSample]:
    """Deterministic per-episode tick-prefix subset (for A_small).

    Keeps the first ``ceil(fraction * n_ticks)`` samples of each
    episode (sorted by tick). No randomness — repeated calls return
    the identical subset.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    by_episode: dict[str, list[TransitionSample]] = {}
    for s in samples:
        by_episode.setdefault(s.episode_id, []).append(s)
    out: list[TransitionSample] = []
    for ep_id in sorted(by_episode.keys()):
        ep_samples = sorted(by_episode[ep_id], key=lambda s: s.tick)
        keep = max(1, math.ceil(fraction * len(ep_samples)))
        out.extend(ep_samples[:keep])
    return out


def downsample_matched(
    samples: Sequence[TransitionSample],
    n: int,
    *,
    seed: int,
) -> list[TransitionSample]:
    """Deterministic seeded subsample of ``samples`` to exactly ``n`` rows.

    Used for D_matched: downsample D to |C|. Raises if ``n`` exceeds the
    available sample count (matching must never upsample). Order of the
    result is stable (original order preserved).
    """
    pool = list(samples)
    if n > len(pool):
        raise ValueError(
            f"cannot downsample {len(pool)} samples to {n} (n > available)"
        )
    if n == len(pool):
        return pool
    rng = random.Random(seed)
    chosen_idx = sorted(rng.sample(range(len(pool)), n))
    return [pool[i] for i in chosen_idx]


def build_treatments(
    main_train_samples: Sequence[TransitionSample],
    random_train_samples: Sequence[TransitionSample],
    *,
    small_fraction: float = 0.25,
    match_seed: int = 20260729,
) -> dict[str, list[TransitionSample]]:
    """Construct the five treatment training sets.

    Parameters
    ----------
    main_train_samples:
        TRAIN-split samples of the multi-policy + branches dataset
        (conventional + counterfactual rows mixed; branch rows are
        detected via :func:`is_counterfactual_sample`).
    random_train_samples:
        TRAIN-split samples of the RandomPolicy-only dataset (must be
        conventional only — asserted).
    small_fraction:
        Tick-prefix fraction for A_small.
    match_seed:
        RNG seed for the deterministic D→D_matched downsample.

    Returns
    -------
    dict[str, list[TransitionSample]]
        Treatment name → training samples. Invariants (asserted):
        ``len(D_matched) == len(C_transition)``; C contains no branch
        rows; D == C + branch rows.
    """
    conventional = [
        s for s in main_train_samples if not is_counterfactual_sample(s)
    ]
    branch_rows = [
        s for s in main_train_samples if is_counterfactual_sample(s)
    ]
    for s in random_train_samples:
        if is_counterfactual_sample(s):
            raise ValueError(
                f"B_random dataset unexpectedly contains a counterfactual "
                f"sample (episode {s.episode_id!r})"
            )

    treatment_c = conventional
    treatment_d = conventional + branch_rows
    treatment_d_matched = downsample_matched(
        treatment_d, len(treatment_c), seed=match_seed
    )
    treatments = {
        "A_small": prefix_subset(conventional, small_fraction),
        "B_random": list(random_train_samples),
        "C_transition": treatment_c,
        "D_counterfactual": treatment_d,
        "D_matched": treatment_d_matched,
    }
    assert len(treatments["D_matched"]) == len(treatments["C_transition"]), (
        "D_matched must have exactly as many samples as C_transition"
    )
    return treatments


@dataclass(frozen=True)
class PairedStats:
    """Paired-difference statistics for one comparison (X vs Y).

    ``diffs[i] = metric(X, run_i) - metric(Y, run_i)`` over paired runs
    (same train seed, same model seed).
    """

    mean_diff: float
    ci_low: float
    ci_high: float
    direction_consistency: float  # fraction of pairs agreeing with sign(mean)
    n_pairs: int
    diffs: tuple[float, ...]

    def to_dict(self) -> dict:
        return {
            "mean_diff": self.mean_diff,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "direction_consistency": self.direction_consistency,
            "n_pairs": self.n_pairs,
            "diffs": list(self.diffs),
        }


def paired_stats(
    diffs: Sequence[float],
    *,
    n_boot: int = 10000,
    seed: int = 0,
    ci: float = 0.95,
) -> PairedStats:
    """Mean paired diff + percentile bootstrap CI + sign consistency.

    NaN diffs are dropped (with the drop reflected in ``n_pairs``).
    With < 2 valid pairs the CI collapses to the point estimate.
    """
    arr = np.asarray([d for d in diffs if d == d], dtype=np.float64)
    n = arr.size
    if n == 0:
        return PairedStats(
            mean_diff=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            direction_consistency=float("nan"),
            n_pairs=0,
            diffs=(),
        )
    mean_diff = float(np.mean(arr))
    if n == 1:
        ci_low = ci_high = mean_diff
    else:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=(n_boot, n))
        boot_means = arr[idx].mean(axis=1)
        alpha = (1.0 - ci) / 2.0
        ci_low = float(np.quantile(boot_means, alpha))
        ci_high = float(np.quantile(boot_means, 1.0 - alpha))
    if mean_diff > 0:
        agree = float(np.mean(arr > 0))
    elif mean_diff < 0:
        agree = float(np.mean(arr < 0))
    else:
        agree = float(np.mean(arr == 0))
    return PairedStats(
        mean_diff=mean_diff,
        ci_low=ci_low,
        ci_high=ci_high,
        direction_consistency=agree,
        n_pairs=int(n),
        diffs=tuple(float(d) for d in arr),
    )


def aggregate_model_seed_diffs(
    diffs_by_run: Mapping[tuple[int, int], float],
) -> tuple[dict[int, float], dict[int, dict[str, float | int]]]:
    """Aggregate paired differences by independent data seed.

    Parameters
    ----------
    diffs_by_run:
        Mapping ``(data_seed, model_seed) -> paired difference``.

    Returns
    -------
    seed_means:
        One mean difference per independent data seed.  These values are
        the statistical units passed to :func:`paired_stats`.
    diagnostics:
        Within-data-seed model-repeat count, mean, standard deviation,
        minimum and maximum.  These quantify model-seed stability only.
    """

    grouped: dict[int, list[float]] = {}
    for (data_seed, _model_seed), diff in diffs_by_run.items():
        value = float(diff)
        if math.isfinite(value):
            grouped.setdefault(int(data_seed), []).append(value)

    seed_means: dict[int, float] = {}
    diagnostics: dict[int, dict[str, float | int]] = {}
    for data_seed in sorted(grouped):
        values = np.asarray(grouped[data_seed], dtype=np.float64)
        if values.size == 0:
            continue
        mean = float(np.mean(values))
        seed_means[data_seed] = mean
        diagnostics[data_seed] = {
            "n_model_seeds": int(values.size),
            "mean": mean,
            "std": (
                float(np.std(values, ddof=1))
                if values.size > 1
                else 0.0
            ),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return seed_means, diagnostics


def verdict_from_stats(
    stats: PairedStats,
    *,
    higher_is_better: bool = True,
    null_epsilon: float = NULL_EPSILON,
) -> str:
    """Engineering-pilot verdict: PASS / NEGATIVE / NULL / INCONCLUSIVE.

    - PASS: the bootstrap CI excludes 0 in the favorable direction.
    - NEGATIVE: the CI excludes 0 in the unfavorable direction.
    - NULL: the CI spans 0 AND |mean diff| < ``null_epsilon``.
    - INCONCLUSIVE: the CI spans 0 with |mean diff| >= ``null_epsilon``.

    NOT a confirmatory verdict — pilot scale, single scenario, single
    task protocol. Qualifiers are attached by the caller.
    """
    if stats.n_pairs == 0 or stats.mean_diff != stats.mean_diff:
        return "INCONCLUSIVE"
    lo, hi = stats.ci_low, stats.ci_high
    if not higher_is_better:
        lo, hi = -hi, -lo
    if lo > 0:
        return "PASS"
    if hi < 0:
        return "NEGATIVE"
    if abs(stats.mean_diff) < null_epsilon:
        return "NULL"
    return "INCONCLUSIVE"
