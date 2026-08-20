"""Gymnasium adapter subpackage (A-05).

Exports the :class:`GymnasiumAdapter` and capability helpers.
"""

from .adapter import GymnasiumAdapter, make_cartpole_env
from .capability import (
    GYMNASIUM_WORLD_ID,
    GYMNASIUM_WORLD_VERSION,
    make_gymnasium_discrete_capability,
)

__all__ = [
    "GymnasiumAdapter",
    "make_cartpole_env",
    "make_gymnasium_discrete_capability",
    "GYMNASIUM_WORLD_ID",
    "GYMNASIUM_WORLD_VERSION",
]
