#!/usr/bin/env bash
#
# Agent Teams — one-command installer.
#
# Checks your machine, installs the teams the best local way, and gets a
# provider configured — adopting an existing Hermes (~/.hermes) if you have one,
# otherwise launching `hermes setup` (which also covers custom / OpenAI-compatible
# endpoints). Idempotent: safe to re-run.
#
# On a fresh machine it auto-installs missing OS prerequisites (git, Python's
# venv module) via the system package manager — using sudo only when that needs
# no password; otherwise it prints the exact command and stops. It never removes
# or replaces your system Python.
#
# Works two ways:
#   • from a clone:   bash install.sh
#   • from the web:   bash <(curl -fsSL <raw-url>/install.sh)     # clones first
# When run from the web it clones into ./agent-teams (override: AGENT_TEAMS_DIR).
#
# Usage:
#   bash install.sh [--no-run] [--no-setup] [--no-browser] [--yes]
#     --no-run      install only; don't offer to start the dashboard at the end
#     --no-setup    don't launch `hermes setup` even if no provider is found
#     --no-browser  skip the Chromium download (browser tools won't work)
#     --yes, -y     non-interactive: never prompt (also skips the setup wizard)
#     --help, -h    show this help
#
# Hands-free provider config (AI-agent / CI installs) — set before running:
#   TEAMS_SETUP_MODEL=deepseek-chat  TEAMS_SETUP_BASE_URL=http://localhost:4000/v1
#   TEAMS_SETUP_PROVIDER=custom      TEAMS_SETUP_API_KEY=sk-...
#   (only TEAMS_SETUP_MODEL is required; the installer then skips the interactive wizard)
#
# Prefer containers? Use Docker instead:  docker compose up --build
#
set -euo pipefail

usage() { sed -n '2,33p' "$0" | sed 's/^#\{0,1\} \{0,1\}//'; }

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

# ---- system packages (git, python venv, curl/wget) on a fresh machine -------
# A clean OS often lacks git, Python's venv module, curl, or wget. Auto-install
# them via the OS package manager when we can do so unattended (root, or
# passwordless sudo) — otherwise print the exact command and stop. We never run
# a sudo that could block on a password prompt (that would hang a non-interactive
# install).
_PKG=""; _PKG_UPDATE=""        # filled by _pkg_setup; empty ⇒ can't auto-install
_pkg_setup() {
  local s=""
  if [ "$(id -u)" -ne 0 ]; then
    if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then s="sudo "; else return 0; fi
  fi
  if   command -v apt-get >/dev/null 2>&1; then _PKG="${s}apt-get install -y -q"; _PKG_UPDATE="${s}apt-get update -qq"
  elif command -v dnf     >/dev/null 2>&1; then _PKG="${s}dnf install -y"
  elif command -v yum     >/dev/null 2>&1; then _PKG="${s}yum install -y"
  elif command -v pacman  >/dev/null 2>&1; then _PKG="${s}pacman -S --noconfirm"
  elif command -v zypper  >/dev/null 2>&1; then _PKG="${s}zypper --non-interactive install"
  elif command -v apk     >/dev/null 2>&1; then _PKG="${s}apk add"
  elif command -v brew    >/dev/null 2>&1; then _PKG="brew install"
  fi
}
_pkg_setup

_pkg_install() {
  # _pkg_install <pkg> [pkg2 ...] — install OS packages if possible
  [ -z "$_PKG" ] && return 1
  [ -n "$_PKG_UPDATE" ] && $_PKG_UPDATE >/dev/null 2>&1 || true
  $_PKG "$@" >/dev/null 2>&1
}

ensure_git() {
  command -v git >/dev/null 2>&1 && return 0
  if [ -n "$_PKG" ]; then
    info "git not found — installing it…"
    _pkg_install git || true
  fi
  command -v git >/dev/null 2>&1 && { info "git installed."; return 0; }
  die "git is required to fetch agent-teams. Install it (e.g. sudo apt install git) and retry."
}

ensure_curl() {
  command -v curl >/dev/null 2>&1 && return 0
  command -v wget >/dev/null 2>&1 && return 0
  if [ -n "$_PKG" ]; then
    info "curl/wget not found — installing curl…"
    _pkg_install curl || true
  fi
  command -v curl >/dev/null 2>&1 && { info "curl installed."; return 0; }
  warn "Neither curl nor wget available — pip bootstrap may fail if needed."
}

# Ensure "$PYBIN -m venv" works (Debian/Ubuntu split it into python3-venv).
# Probe by actually creating a throwaway venv: on Debian `import ensurepip`
# succeeds even when the package is missing — it's the pip bootstrap that fails.
_venv_ok() { local p="${TMPDIR:-/tmp}/.hs_venvprobe.$$"; rm -rf "$p"
  "$PYBIN" -m venv "$p" >/dev/null 2>&1; local r=$?; rm -rf "$p"; return $r; }
ensure_py_venv() {
  _venv_ok && return 0
  local pyver pkg
  pyver="$("$PYBIN" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo 3)"
  if [ -n "$_PKG" ]; then
    [ -n "$_PKG_UPDATE" ] && $_PKG_UPDATE >/dev/null 2>&1 || true
    for pkg in "python${pyver}-venv" python3-venv; do
      info "Python venv module missing — installing $pkg…"
      $_PKG "$pkg" >/dev/null 2>&1 || continue
      _venv_ok && { info "venv module ready."; return 0; }
    done
  fi
  die "Python's venv module is unavailable. Install it (e.g. sudo apt install python3-venv) and retry — or use Docker: docker compose up --build"
}

# ---- Locate or fetch the repo --------------------------------------------
# Run from a checkout → use it. Piped from the web (no checkout) → clone first,
# then continue from inside the clone.
REPO_URL="${AGENT_TEAMS_REPO:-https://github.com/CyberTron957/agent-teams.git}"
_in_repo() { [ -f "$1/pyproject.toml" ] && grep -q 'name *= *"agent-teams"' "$1/pyproject.toml" 2>/dev/null; }

_self_dir=""
case "$0" in */*) _self_dir="$(cd "$(dirname "$0")" 2>/dev/null && pwd)" ;; esac
if [ -n "$_self_dir" ] && _in_repo "$_self_dir"; then
  cd "$_self_dir"                                  # bash install.sh from a clone
elif _in_repo "$PWD"; then
  :                                                # already inside a checkout
else
  ensure_git                                        # auto-installs git on a fresh machine
  ensure_curl                                       # auto-installs curl on a fresh machine
  TARGET="${AGENT_TEAMS_DIR:-$PWD/agent-teams}"
  if _in_repo "$TARGET"; then
    step "Updating existing clone at $TARGET"
    git -C "$TARGET" pull --ff-only 2>/dev/null || warn "couldn't fast-forward; using the existing clone."
  else
    step "Fetching agent-teams → $TARGET"
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
  for c in python3.14 python3.13 python3.12 python3.11 python3 python; do
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
  if [ ! -d "$VENV" ]; then
    ensure_py_venv                                  # auto-installs python3-venv if missing
    "$PYBIN" -m venv "$VENV"
  fi
  info "venv: $VENV"
fi
PY="$VENV/bin/python"
[ -x "$PY" ] || die "venv python missing at $PY"

# Ensure the active virtual environment's python is 3.11+
if ! "$PY" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 11) else 1)' 2>/dev/null; then
  die "The active virtual environment ($VENV) uses Python $("$PY" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))' 2>/dev/null || echo 'unknown'), but Python 3.11+ is required."
fi

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
step "Installing agent-teams (+ hermes-agent) — this can take a few minutes"
"$PY" -m pip install --quiet --upgrade pip
if [ -f pyproject.toml ] && grep -q 'name *= *"agent-teams"' pyproject.toml; then
  info "Editable install from this repo (pip install -e .)"
  "$PY" -m pip install -e .          # editable → no stale shadow copy
else
  info "Installing from PyPI (pip install agent-teams)"
  "$PY" -m pip install agent-teams
fi
HV="$("$PY" -c 'from importlib.metadata import version; print(version("hermes-agent"))' 2>/dev/null || echo '?')"
info "hermes-agent $HV installed."

# ---- 4. browser for the publishing tools ---------------------------------
if [ "$NO_BROWSER" -eq 0 ]; then
  step "Browser (for the browser/publishing tools)"
  if "$PY" -c 'import sys; from teams_server.browser_pool import _find_browser; sys.exit(0 if _find_browser() else 1)' 2>/dev/null; then
    info "Found a usable Chrome/Chromium — skipping the download."
  else
    # Try installing system Chromium first (fastest, no download needed)
    _chromium_installed=0
    if command -v chromium-browser >/dev/null 2>&1 || command -v chromium >/dev/null 2>&1; then
      info "System Chromium found — using it."
      _chromium_installed=1
    elif [ -n "$_PKG" ]; then
      info "No system Chrome found — attempting to install chromium via package manager…"
      _pkg_install chromium-browser chromium 2>/dev/null && _chromium_installed=1 || true
      if [ "$_chromium_installed" -eq 1 ]; then
        info "System Chromium installed."
      fi
    fi

    # Fall back to Playwright Chromium download
    if [ "$_chromium_installed" -eq 0 ]; then
      # The `playwright` Python package isn't always pulled in by hermes-agent[all];
      # make sure it's importable before we ask it to fetch a browser.
      if ! "$PY" -c 'import playwright' 2>/dev/null; then
        info "Installing the playwright package…"
        "$PY" -m pip install --quiet playwright || warn "couldn't install playwright."
      fi
      info "Downloading Playwright Chromium (with system dependencies)…"
      # --with-deps installs OS libraries Chromium needs (libnss, libatk, etc.)
      # If --with-deps fails (e.g. broken repos), try without deps as last resort
      if ! "$PY" -m playwright install --with-deps chromium 2>/dev/null; then
        warn "Playwright --with-deps failed; trying without system deps…"
        "$PY" -m playwright install chromium \
          || warn "Chromium install failed; browser tools will be unavailable (everything else works)."
      fi
    fi
  fi
else
  warn "--no-browser: skipping Chromium (browser tools will be unavailable)."
fi

# ---- 5. provider / model (adopt existing, else set up) -------------------
step "Provider / model"
is_configured() { "$PY" -c 'import sys; from teams_server.model_config import is_model_configured; sys.exit(0 if is_model_configured() else 1)' 2>/dev/null; }
if [ -n "${TEAMS_SETUP_MODEL:-}" ]; then
  # Non-interactive config for AI-agent / CI / headless installs. Set:
  #   TEAMS_SETUP_MODEL=...  [TEAMS_SETUP_PROVIDER=...]  [TEAMS_SETUP_BASE_URL=...]  [TEAMS_SETUP_API_KEY=...]
  info "Configuring provider from TEAMS_SETUP_* env (non-interactive)…"
  "$VENV/bin/agent-teams" set-model \
    ${TEAMS_SETUP_PROVIDER:+--provider "$TEAMS_SETUP_PROVIDER"} \
    --model "$TEAMS_SETUP_MODEL" \
    ${TEAMS_SETUP_BASE_URL:+--base-url "$TEAMS_SETUP_BASE_URL"} \
    ${TEAMS_SETUP_API_KEY:+--api-key "$TEAMS_SETUP_API_KEY"} \
    || warn "set-model failed — configure later with: $VENV/bin/agent-teams set-model --help"
elif is_configured; then
  # Detect WHICH home already has a model so the message is honest about what we
  # adopted (a personal `hermes setup` in ~/.hermes, or a prior teams default).
  _src="$("$PY" -c 'from teams_server.model_config import get_default_model, detect_global_hermes_model as g; import sys; sys.stdout.write("teams default (data/.hermes-shared)" if get_default_model().get("model") else ("your existing ~/.hermes setup" if g().get("model") else "?"))' 2>/dev/null || echo '?')"
  info "Provider already configured — adopting $_src. Nothing to do."
  info "(Reconfigure or add tool keys anytime with: $VENV/bin/agent-teams setup)"
elif [ "$NO_SETUP" -eq 1 ] || [ "$ASSUME_YES" -eq 1 ] || [ ! -t 0 ]; then
  warn "No provider configured. Set one non-interactively:  $VENV/bin/agent-teams set-model --model <m> [--base-url <url>] [--api-key <k>]"
  warn "…or run the interactive wizard:  $VENV/bin/hermes setup"
else
  info "No provider configured — launching Hermes' setup wizard (40+ providers)…"
  "$VENV/bin/hermes" setup || warn "hermes setup didn't finish — rerun: $VENV/bin/hermes setup"
fi

# ---- 6. verify ------------------------------------------------------------
step "Verifying (agent-teams doctor)"
"$VENV/bin/agent-teams" doctor || warn "doctor flagged issues above — resolve them before starting."

# ---- 7. scaffold a starter team (no-op-safe) ------------------------------
step "Scaffolding a starter team"
"$VENV/bin/agent-teams" init || warn "init skipped (see above)."

# ---- 8. expose binaries to user PATH --------------------------------------
step "Exposing commands to user PATH"
if [ -d "$HOME/.local/bin" ] || mkdir -p "$HOME/.local/bin" 2>/dev/null; then
  ln -sf "$VENV/bin/agent-teams" "$HOME/.local/bin/agent-teams"
  ln -sf "$VENV/bin/hermes" "$HOME/.local/bin/hermes"
  info "Symlinked agent-teams and hermes to $HOME/.local/bin"

  # Ensure $HOME/.local/bin is in PATH in ~/.bashrc, ~/.bash_profile, ~/.zshrc, or ~/.profile if it's not already
  for f in "$HOME/.bashrc" "$HOME/.bash_profile" "$HOME/.zshrc" "$HOME/.profile"; do
    if [ -f "$f" ]; then
      if ! grep -q '\.local/bin' "$f" 2>/dev/null; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$f"
        info "Added $HOME/.local/bin to $f"
      fi
    fi
  done
else
  warn "Could not create $HOME/.local/bin to symlink commands."
fi

# ---- done: start now, or print the one command to do it -------------------
# Absolute paths so they work from any shell (e.g. after a web-bootstrap clone).
START_CMD="$VENV/bin/agent-teams up"
printf "%s" "$G"
cat << "EOF"

    _                    _     _____
   / \   __ _  ___ _ __ | |_  |_   _|___  __ _ _ __ ___  ___
  / _ \ / _` |/ _ \ '_ \| __|   | |/ _ \/ _` | '_ ` _ \/ __|
 / ___ \ (_| |  __/ | | | |_    | |  __/ (_| | | | | | \__ \
/_/   \_\__, |\___|_| |_|\__|   |_|\___|\__,_|_| |_| |_|___/
         |___/

    Commands:
      • Start:  agent-teams up
      • Stop:        agent-teams down
      • Status:      agent-teams status
      • Config:      agent-teams setup
      • Doctor:      agent-teams doctor

EOF
printf "%s" "$X"
step "Done — your teams is installed in $PWD"
info "(If commands are not found yet in this terminal, run 'source ~/.bashrc' or use the full path: $START_CMD)"
if [ "$NO_RUN" -eq 0 ] && [ "$ASSUME_YES" -eq 0 ] && [ -t 0 ]; then
  printf '\n    Start it now? [Y/n] '
  read -r _ans
  case "${_ans:-y}" in
    [Nn]*) info "Not started — run the command above when you're ready." ;;
    *)     step "Starting (Ctrl-C to stop)"; exec "$VENV/bin/agent-teams" up ;;
  esac
fi
