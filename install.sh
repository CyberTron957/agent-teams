#!/usr/bin/env bash
#
# Hermes Swarm — one-command installer.
#
# Checks your machine, installs the swarm the best local way, and gets a
# provider configured — adopting an existing Hermes (~/.hermes) if you have one,
# otherwise launching `hermes setup` (which also covers custom / OpenAI-compatible
# endpoints). Idempotent: safe to re-run.
#
# Works two ways:
#   • from a clone:   bash install.sh
#   • from the web:   bash <(curl -fsSL <raw-url>/install.sh)     # clones first
# When run from the web it clones into ./hermes-swarm (override: HERMES_SWARM_DIR).
#
# Usage:
#   bash install.sh [--no-run] [--no-setup] [--no-browser] [--yes]
#     --no-run      install only; don't offer to start the dashboard at the end
#     --no-setup    don't launch `hermes setup` even if no provider is found
#     --no-browser  skip the Chromium download (browser tools won't work)
#     --yes, -y     non-interactive: never prompt (also skips the setup wizard)
#     --help, -h    show this help
#
# Prefer containers? Use Docker instead:  docker compose up --build
#
set -euo pipefail

usage() { sed -n '2,23p' "$0" | sed 's/^#\{0,1\} \{0,1\}//'; }

# ---- pretty output --------------------------------------------------------
if [ -t 1 ]; then B=$'\e[1m'; G=$'\e[32m'; Y=$'\e[33m'; R=$'\e[31m'; X=$'\e[0m'; else B= G= Y= R= X=; fi
step() { printf '\n%s==>%s %s%s%s\n' "$B$G" "$X" "$B" "$1" "$X"; }
info() { printf '    %s\n' "$1"; }
warn() { printf '%s !! %s%s\n' "$Y" "$1" "$X"; }
die()  { printf '%s xx %s%s\n' "$R" "$1" "$X" >&2; exit 1; }

NO_RUN=0; NO_SETUP=0; NO_BROWSER=0; ASSUME_YES=0
for a in "$@"; do case "$a" in
  --no-run)     NO_RUN=1 ;;
  --no-setup)   NO_SETUP=1 ;;
  --no-browser) NO_BROWSER=1 ;;
  --yes|-y)     ASSUME_YES=1 ;;
  -h|--help)    usage; exit 0 ;;
  *)            die "unknown option: $a  (try --help)" ;;
esac; done

# ---- Locate or fetch the repo --------------------------------------------
# Run from a checkout → use it. Piped from the web (no checkout) → clone first,
# then continue from inside the clone.
REPO_URL="${HERMES_SWARM_REPO:-https://github.com/CyberTron957/hermes-mission-control.git}"
_in_repo() { [ -f "$1/pyproject.toml" ] && grep -q 'name *= *"hermes-swarm"' "$1/pyproject.toml" 2>/dev/null; }

_self_dir=""
case "$0" in */*) _self_dir="$(cd "$(dirname "$0")" 2>/dev/null && pwd)" ;; esac
if [ -n "$_self_dir" ] && _in_repo "$_self_dir"; then
  cd "$_self_dir"                                  # bash install.sh from a clone
elif _in_repo "$PWD"; then
  :                                                # already inside a checkout
else
  command -v git >/dev/null 2>&1 || die "git is required to fetch hermes-swarm — install it and retry."
  TARGET="${HERMES_SWARM_DIR:-$PWD/hermes-swarm}"
  if _in_repo "$TARGET"; then
    step "Updating existing clone at $TARGET"
    git -C "$TARGET" pull --ff-only 2>/dev/null || warn "couldn't fast-forward; using the existing clone."
  else
    step "Fetching hermes-swarm → $TARGET"
    git clone --depth 1 "$REPO_URL" "$TARGET" || die "git clone failed ($REPO_URL)."
  fi
  cd "$TARGET"
fi

# ---- 1. OS + Python 3.11+ -------------------------------------------------
step "Checking prerequisites"
case "$(uname -s)" in
  Linux|Darwin) : ;;
  *) warn "Untested OS '$(uname -s)' — proceeding, but Docker may be smoother." ;;
esac

find_python() {
  for c in python3.13 python3.12 python3.11 python3 python; do
    command -v "$c" >/dev/null 2>&1 || continue
    "$c" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 11) else 1)' 2>/dev/null \
      && { command -v "$c"; return 0; }
  done
  return 1
}
PYBIN="$(find_python)" || die "Python 3.11+ not found. Install it, or use Docker: docker compose up --build"
info "Python: $("$PYBIN" --version 2>&1)  ($PYBIN)"

# ---- 2. virtual environment ----------------------------------------------
step "Setting up the virtual environment"
if [ -n "${VIRTUAL_ENV:-}" ]; then
  VENV="$VIRTUAL_ENV"; info "Using the active venv: $VENV"
else
  VENV="$PWD/.venv"
  [ -d "$VENV" ] || "$PYBIN" -m venv "$VENV"
  info "venv: $VENV"
fi
PY="$VENV/bin/python"
[ -x "$PY" ] || die "venv python missing at $PY"

# Ensure pip is installed in the virtual environment
if ! "$PY" -m pip --version >/dev/null 2>&1; then
  info "pip not found in venv; attempting to bootstrap it..."
  if ! "$PY" -m ensurepip --default-pip >/dev/null 2>&1; then
    info "ensurepip failed; downloading get-pip.py to bootstrap pip..."
    if command -v curl >/dev/null 2>&1; then
      curl -sSL https://bootstrap.pypa.io/get-pip.py | "$PY" >/dev/null 2>&1 || warn "Failed to bootstrap pip using curl."
    elif command -v wget >/dev/null 2>&1; then
      wget -qO- https://bootstrap.pypa.io/get-pip.py | "$PY" >/dev/null 2>&1 || warn "Failed to bootstrap pip using wget."
    else
      warn "Neither curl nor wget found; could not bootstrap pip."
    fi
  fi
fi

# ---- 3. install the package ----------------------------------------------
step "Installing hermes-swarm (+ hermes-agent) — this can take a few minutes"
"$PY" -m pip install --quiet --upgrade pip
if [ -f pyproject.toml ] && grep -q 'name *= *"hermes-swarm"' pyproject.toml; then
  info "Editable install from this repo (pip install -e .)"
  "$PY" -m pip install -e .          # editable → no stale shadow copy
else
  info "Installing from PyPI (pip install hermes-swarm)"
  "$PY" -m pip install hermes-swarm
fi
HV="$("$PY" -c 'from importlib.metadata import version; print(version("hermes-agent"))' 2>/dev/null || echo '?')"
info "hermes-agent $HV installed."

# ---- 4. browser for the publishing tools ---------------------------------
if [ "$NO_BROWSER" -eq 0 ]; then
  step "Browser (for the browser/publishing tools)"
  if "$PY" -c 'import sys; from swarm_server.browser_pool import _find_browser; sys.exit(0 if _find_browser() else 1)' 2>/dev/null; then
    info "Found a usable Chrome/Chromium — skipping the download."
  else
    info "No system Chrome found — downloading Playwright Chromium…"
    "$PY" -m playwright install chromium || warn "Chromium install failed; browser tools will be unavailable."
  fi
else
  warn "--no-browser: skipping Chromium (browser tools will be unavailable)."
fi

# ---- 5. provider / model (adopt existing, else set up) -------------------
step "Provider / model"
is_configured() { "$PY" -c 'import sys; from swarm_server.model_config import is_model_configured; sys.exit(0 if is_model_configured() else 1)' 2>/dev/null; }
if is_configured; then
  info "A provider is already configured (your ~/.hermes or a swarm default) — adopting it. Nothing to do."
elif [ "$NO_SETUP" -eq 1 ] || [ "$ASSUME_YES" -eq 1 ] || [ ! -t 0 ]; then
  warn "No provider configured. Run it yourself when ready:  $VENV/bin/hermes setup"
else
  info "No provider configured — launching Hermes' setup wizard (40+ providers)…"
  "$VENV/bin/hermes" setup || warn "hermes setup didn't finish — rerun: $VENV/bin/hermes setup"
fi

# ---- 6. verify ------------------------------------------------------------
step "Verifying (hermes-swarm doctor)"
"$VENV/bin/hermes-swarm" doctor || warn "doctor flagged issues above — resolve them before starting."

# ---- 7. scaffold a starter team (no-op-safe) ------------------------------
step "Scaffolding a starter team"
"$VENV/bin/hermes-swarm" init || warn "init skipped (see above)."

# ---- done: start now, or print the one command to do it -------------------
# Absolute path so it works from any shell (e.g. after a web-bootstrap clone).
START_CMD="$VENV/bin/hermes-swarm up"
step "Done — your swarm is installed in $PWD"
info "Start the dashboard:  $START_CMD   → http://127.0.0.1:8000"
if [ "$NO_RUN" -eq 0 ] && [ "$ASSUME_YES" -eq 0 ] && [ -t 0 ]; then
  printf '\n    Start it now? [Y/n] '
  read -r _ans
  case "${_ans:-y}" in
    [Nn]*) info "Not started — run the command above when you're ready." ;;
    *)     step "Starting (Ctrl-C to stop)"; exec "$VENV/bin/hermes-swarm" up ;;
  esac
fi
