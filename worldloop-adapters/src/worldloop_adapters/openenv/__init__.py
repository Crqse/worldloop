"""OpenEnv adapter subpackage (A-06/A-07).

Exports:
- :class:`OpenEnvWorldAdapter`: wraps an OpenEnv client (or any client
  satisfying the OpenEnv reset/step/state protocol) as a kernel
  :class:`WorldProtocol`.
- :class:`OpenEnvServerWrapper`: wraps a kernel world as an OpenEnv-style
  server (in-process; for production HTTP/Docker, use OpenEnv upstream).
- :class:`InProcessOpenEnvClient`: mock OpenEnv client for tests.

OpenEnv is an early-stage project; the adapter does NOT hard-depend on
the ``openenv`` PyPI package. It uses duck typing: any client with
``reset(seed)`` / ``step(action)`` / ``state()`` / ``action_space()``
methods works. Install ``openenv`` only when integrating with the real
OpenEnv training stack.
"""

from .adapter import OpenEnvWorldAdapter, InProcessOpenEnvClient
from .server_wrapper import OpenEnvServerWrapper
from .capability import (
    OPENENV_WORLD_ID,
    OPENENV_WORLD_VERSION,
    make_openenv_capability,
)

__all__ = [
    "OpenEnvWorldAdapter",
    "InProcessOpenEnvClient",
    "OpenEnvServerWrapper",
    "make_openenv_capability",
    "OPENENV_WORLD_ID",
    "OPENENV_WORLD_VERSION",
]
