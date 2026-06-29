"""DEPRECATED import alias: ``swarm_server`` was renamed to ``teams_server``
(the hermes-swarm → agent-teams rename).

This shim exists ONLY so existing installs keep working across the transition:

  * the pre-rename startup auto-update re-execs ``python -m swarm_server.cli``
    (that path is hard-coded in the already-deployed code) — without this shim
    that re-exec would ``ModuleNotFoundError`` and the server would not restart;
  * user scripts / systemd units that ``import swarm_server`` keep resolving.

It aliases the whole package to ``teams_server`` via ``sys.modules`` so every
``swarm_server.X`` is the SAME object as ``teams_server.X`` (no duplicate state),
and emits a one-time deprecation notice. New installs never import this module,
so only existing users — the ones still using the old name — are ever told.

TO REMOVE in a future major version: delete this ``swarm_server/`` directory and
drop it from ``[tool.setuptools] packages`` + the ``hermes-swarm`` console script
in ``pyproject.toml``.
"""

import sys as _sys
import warnings as _warnings

import teams_server as _teams_server

# Machine-readable signal for tooling/tests; human-readable line for operators.
_warnings.warn(
    "'swarm_server' was renamed to 'teams_server' (hermes-swarm → agent-teams). "
    "Update imports / commands; this alias will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)
print(
    "⚠ 'swarm_server' is deprecated and now an alias for 'teams_server' "
    "(hermes-swarm → agent-teams). Update your imports — this alias will be "
    "removed in a future release.",
    file=_sys.stderr,
)

# Make `import swarm_server[.submodule]` resolve to the real package, sharing a
# single set of module objects (config locks, registries, etc. stay singletons).
_sys.modules[__name__] = _teams_server
