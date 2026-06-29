"""Agent Teams — P2P Multi-Agent Framework with Real-Time Monitoring.

Usage:
    agent-teams up                 # after `pip install` (recommended)
    python -m swarm_server          # equivalent module entry point

Hermes is resolved automatically (pip `hermes-agent`, else HERMES_AGENT_PATH,
else ~/.hermes/hermes-agent). See config.ensure_hermes_importable.
"""

# Single source of truth is pyproject.toml; read it back from the installed
# package metadata so the version never drifts between the two.
from importlib.metadata import version as _version, PackageNotFoundError as _PNF

try:
    __version__ = _version("agent-teams")
except _PNF:  # running from a source tree without an install
    __version__ = "0.0.0+source"
