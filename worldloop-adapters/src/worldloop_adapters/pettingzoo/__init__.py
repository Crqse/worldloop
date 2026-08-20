"""PettingZoo Parallel adapter subpackage (A-01).

Exports the :class:`PettingZooParallelAdapter`, the env factories
(:func:`make_simple_spread_env` / :func:`make_simple_tag_env`), and the
Phase 5 exact-restore allowlist helpers.
"""

from .adapter import (
    PettingZooParallelAdapter,
    make_simple_spread_env,
    make_simple_tag_env,
)
from .capability import (
    EXACT_RESTORE_VERIFIED_ENV_FAMILIES,
    PETTINGZOO_WORLD_ID,
    PETTINGZOO_WORLD_VERSION,
    is_exact_restore_verified,
    make_pettingzoo_capability,
    make_pettingzoo_mpe_capability,
    verify_immediate_restore,
)

__all__ = [
    "PettingZooParallelAdapter",
    "make_simple_spread_env",
    "make_simple_tag_env",
    "make_pettingzoo_mpe_capability",
    "make_pettingzoo_capability",
    "is_exact_restore_verified",
    "verify_immediate_restore",
    "EXACT_RESTORE_VERIFIED_ENV_FAMILIES",
    "PETTINGZOO_WORLD_ID",
    "PETTINGZOO_WORLD_VERSION",
]
