"""Model configuration: a swarm-wide DEFAULT plus per-agent OVERRIDES, resolved
into the effective backend each agent actually uses.

Hermes-native: the default is stored in Hermes' OWN config (``config.yaml``
``model:`` + the provider's key in ``.env``) using Hermes' ``load_config`` /
``save_config`` / ``save_env_value`` — wrapped in a HERMES_HOME override so we
target a swarm-managed home (``data/.hermes-shared``) instead of the process
default. The provider catalogue comes straight from Hermes' ``PROVIDER_REGISTRY``
(every provider Hermes supports), plus OpenRouter (which Hermes special-cases)
and a Custom OpenAI-compatible option.

Resolution precedence: per-agent override → swarm default → an existing
``~/.hermes`` setup (offered to adopt) → the legacy LiteLLM proxy fallback, so
existing deployments keep running until the operator picks a model.
"""

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from swarm_server.config import (
    DATA_ROOT,
    DEFAULT_MODEL,
    LITELLM_API_BASE,
    LLM_API_KEY,
    ensure_hermes_importable,
)

log = logging.getLogger("swarm.model")

# Swarm-wide default lives here (Hermes config format; on the data volume).
SHARED_HERMES_HOME = DATA_ROOT / ".hermes-shared"
# The user's personal Hermes home (detect an existing setup to offer/adopt).
GLOBAL_HERMES_HOME = Path(
    os.environ.get("SWARM_GLOBAL_HERMES_HOME") or (Path.home() / ".hermes")
)

# Providers NOT in Hermes' registry that we still want to offer (OpenAI-compatible).
_EXTRA_OPENAI_PRESETS = {
    "openrouter": {"label": "OpenRouter", "base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY"},
    "custom": {"label": "Custom (OpenAI-compatible)", "base_url": "", "key_env": "OPENAI_API_KEY"},
}
# Treat these registry/extra providers as OpenAI-compatible "custom" routing
# (provider=custom + base_url). Everything else uses its native Hermes adapter.
_OPENAI_COMPATIBLE = {"openrouter", "custom", "lmstudio", "openai-api", "nvidia",
                      "deepseek", "together", "groq", "novita", "huggingface"}

_presets_cache: Optional[List[Dict[str, Any]]] = None


# ---------------------------------------------------------------------------
# Hermes config access (wrapped to target a specific HERMES_HOME)
# ---------------------------------------------------------------------------
@contextmanager
def _home(home: Path):
    """Run Hermes config calls against ``home`` via the ContextVar override."""
    ensure_hermes_importable()
    token = None
    try:
        from hermes_constants import set_hermes_home_override, reset_hermes_home_override

        home.mkdir(parents=True, exist_ok=True)
        token = set_hermes_home_override(str(home))
        yield
    finally:
        if token is not None:
            try:
                from hermes_constants import reset_hermes_home_override
                reset_hermes_home_override(token)
            except Exception:
                pass


def _provider_key_env(provider: str) -> str:
    """The .env var holding the given provider's credential."""
    if provider in _EXTRA_OPENAI_PRESETS:
        return _EXTRA_OPENAI_PRESETS[provider]["key_env"]
    try:
        ensure_hermes_importable()
        from hermes_cli.auth import PROVIDER_REGISTRY

        pc = PROVIDER_REGISTRY.get(provider)
        if pc and getattr(pc, "api_key_env_vars", ()):
            return pc.api_key_env_vars[0]
    except Exception:
        pass
    return "OPENAI_API_KEY"


def _read_env_value(env_path: Path, key: str) -> str:
    try:
        if key and env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                if k.strip() == key:
                    return v.strip().strip('"').strip("'")
    except Exception:
        pass
    return ""


def build_provider_presets() -> List[Dict[str, Any]]:
    """Provider catalogue for the setup UI, derived from Hermes' registry.

    Includes every key-based (api_key auth) provider Hermes knows, plus
    OpenRouter and Custom. OAuth-only providers are flagged so the UI can point
    the user at the terminal wizard instead of a key field.
    """
    global _presets_cache
    if _presets_cache is not None:
        return _presets_cache
    out: List[Dict[str, Any]] = []
    # OpenRouter first (most common BYO-key gateway), then Custom.
    for pid in ("openrouter",):
        d = _EXTRA_OPENAI_PRESETS[pid]
        out.append({"id": pid, "label": d["label"], "base_url": d["base_url"],
                    "key_env": d["key_env"], "auth_type": "api_key",
                    "openai_compatible": True, "needs_base_url": False, "models": []})
    try:
        ensure_hermes_importable()
        from hermes_cli.auth import PROVIDER_REGISTRY

        try:
            from hermes_cli.setup import _DEFAULT_PROVIDER_MODELS as _PM
        except Exception:
            _PM = {}
        for pid, pc in PROVIDER_REGISTRY.items():
            keys = getattr(pc, "api_key_env_vars", ()) or ()
            out.append({
                "id": pid,
                "label": getattr(pc, "name", pid),
                "base_url": getattr(pc, "inference_base_url", "") or "",
                "key_env": keys[0] if keys else "",
                "auth_type": getattr(pc, "auth_type", "api_key"),
                "openai_compatible": pid in _OPENAI_COMPATIBLE,
                "needs_base_url": False,
                "models": list(_PM.get(pid, []))[:8],
            })
    except Exception as e:
        log.warning("provider registry unavailable (%s) — minimal preset list", e)
    # Custom last.
    d = _EXTRA_OPENAI_PRESETS["custom"]
    out.append({"id": "custom", "label": d["label"], "base_url": "", "key_env": d["key_env"],
                "auth_type": "api_key", "openai_compatible": True, "needs_base_url": True, "models": []})
    _presets_cache = out
    return out


def _preset(provider: str) -> Dict[str, Any]:
    for p in build_provider_presets():
        if p["id"] == provider:
            return p
    return {}


# ---------------------------------------------------------------------------
# Read / write a model choice in a Hermes home
# ---------------------------------------------------------------------------
def read_model_from_home(home: Path) -> Dict[str, Any]:
    """Read {provider, model, base_url, api_key} from a Hermes home (via Hermes)."""
    provider = model = base_url = ""
    try:
        with _home(home):
            from hermes_cli.config import load_config, cfg_get

            cfg = load_config()
        mc = cfg.get("model")
        if isinstance(mc, str):
            model = mc.strip()
        elif isinstance(mc, dict):
            model = str(mc.get("default") or "").strip()
            provider = str(mc.get("provider") or "").strip()
            base_url = str(mc.get("base_url") or "").strip()
    except Exception as e:
        log.debug("read_model_from_home(%s) failed: %s", home, e)
    api_key = _read_env_value(home / ".env", _provider_key_env(provider)) if provider else ""
    return {"provider": provider, "model": model, "base_url": base_url, "api_key": api_key}


def write_model_to_home(home: Path, provider: str, model: str, base_url: str, api_key: str) -> None:
    """Persist the choice using Hermes' own save_config + save_env_value."""
    with _home(home):
        from hermes_cli.config import load_config, save_config, save_env_value

        cfg = load_config()
        mc = cfg.get("model")
        if not isinstance(mc, dict):
            mc = {}
        mc["default"] = model
        mc["provider"] = provider or "custom"
        if base_url:
            mc["base_url"] = base_url
        else:
            mc.pop("base_url", None)
        cfg["model"] = mc
        save_config(cfg)  # Hermes normalizes + refuses secrets in config.yaml
        if api_key:
            save_env_value(_provider_key_env(provider), api_key)  # → home/.env
    log.info("Model written to %s: provider=%s model=%s", home.name, provider, model)


# ---------------------------------------------------------------------------
# High-level: default, detection, resolution
# ---------------------------------------------------------------------------
def get_default_model() -> Dict[str, Any]:
    return read_model_from_home(SHARED_HERMES_HOME)


def set_default_model(provider: str, model: str, base_url: str = "", api_key: str = "") -> None:
    write_model_to_home(SHARED_HERMES_HOME, provider, model, base_url, api_key)


def detect_global_hermes_model() -> Dict[str, Any]:
    return read_model_from_home(GLOBAL_HERMES_HOME)


def is_model_configured() -> bool:
    if get_default_model().get("model"):
        return True
    if os.environ.get("SWARM_LLM_BASE_URL"):
        return True
    return False


def resolve_model(agent_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Effective {provider, model, base_url, api_key, source, display_provider}.

    Per-agent override (layered over) → swarm default → legacy proxy. The
    routing provider is the native Hermes provider id, EXCEPT OpenAI-compatible
    providers (OpenRouter/Custom/etc.) which route via ``custom`` + base_url.
    """
    agent_cfg = agent_cfg or {}
    default = get_default_model()
    if default.get("model"):
        base = dict(default)
        base_source = "default"
    else:
        base = {"provider": "custom", "model": DEFAULT_MODEL,
                "base_url": LITELLM_API_BASE, "api_key": LLM_API_KEY}
        base_source = "proxy"

    ov_model = (agent_cfg.get("model") or "").strip()
    ov_provider = (agent_cfg.get("provider") or "").strip()
    ov_base = (agent_cfg.get("base_url") or "").strip()
    ov_key = (agent_cfg.get("api_key") or "").strip()
    overridden = bool(ov_model or ov_provider or ov_base or ov_key)

    provider = ov_provider or base.get("provider") or "custom"
    model = ov_model or base.get("model") or DEFAULT_MODEL
    api_key = ov_key or base.get("api_key") or ""

    preset = _preset(provider)
    openai_compat = preset.get("openai_compatible", True) if preset else True

    if openai_compat:
        # Custom path: needs a base_url. Inherit/override, else preset, else proxy.
        base_url = ov_base or base.get("base_url") or preset.get("base_url") or ""
        route_provider = "custom"
        if not base_url:
            base_url = LITELLM_API_BASE  # last resort so the call has an endpoint
    else:
        # Native provider: Hermes resolves the endpoint from its registry. Pass a
        # base_url only if one was explicitly set (don't inherit a proxy URL).
        base_url = ov_base if ov_provider else (ov_base or base.get("base_url") or "")
        route_provider = provider

    return {
        "provider": route_provider, "display_provider": provider, "model": model,
        "base_url": base_url, "api_key": api_key,
        "source": "agent" if overridden else base_source,
    }
