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


def _parse_env_file(env_path: Path) -> Dict[str, str]:
    """All KEY=VALUE pairs in a ``.env`` (quotes stripped). Empty on any error."""
    out: Dict[str, str] = {}
    try:
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip()
                if k:
                    out[k] = v.strip().strip('"').strip("'")
    except Exception:
        pass
    return out


def _read_env_value(env_path: Path, key: str) -> str:
    return _parse_env_file(env_path).get(key, "") if key else ""


def import_hermes_secrets() -> List[str]:
    """Load existing Hermes ``.env`` secrets into THIS process's environment, so
    swarm agents inherit every provider/tool API key already configured for
    Hermes — not just the model provider's key.

    Reads two homes: the swarm's shared home (``data/.hermes-shared`` — where
    ``hermes setup`` writes in the Docker flow) takes precedence, then the user's
    personal ``~/.hermes`` (the pip/local flow). Agents run in-process (each under
    its own HERMES_HOME), so values in ``os.environ`` are visible to all of them;
    Hermes' web/tool plugins read them via ``os.getenv``. NON-overriding:
    anything already set in the environment (explicit server/deployment config,
    SWARM_*) wins, so this never clobbers it. Returns the names imported (for
    logging — values are never logged).
    """
    imported: List[str] = []
    for home in (SHARED_HERMES_HOME, GLOBAL_HERMES_HOME):
        for k, v in _parse_env_file(home / ".env").items():
            if k and k not in os.environ:
                os.environ[k] = v
                imported.append(k)
    if imported:
        log.info("Imported %d Hermes secret(s) into the environment: %s",
                 len(imported), ", ".join(sorted(imported)))
    return imported


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
    registry_ok = False
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
                # A provider that ships no default endpoint (e.g. Azure Foundry,
                # which exposes AZURE_FOUNDRY_BASE_URL) REQUIRES the operator to
                # supply one. Flag it so the setup UI shows + requires the base
                # URL field instead of leaving only key + model.
                "needs_base_url": not (getattr(pc, "inference_base_url", "") or ""),
                "models": list(_PM.get(pid, []))[:8],
            })
        registry_ok = True
    except Exception as e:
        log.warning("provider registry unavailable (%s) — minimal preset list", e)
    # Custom last.
    d = _EXTRA_OPENAI_PRESETS["custom"]
    out.append({"id": "custom", "label": d["label"], "base_url": "", "key_env": d["key_env"],
                "auth_type": "api_key", "openai_compatible": True, "needs_base_url": True, "models": []})
    # Only cache when the registry import SUCCEEDED. Caching the degraded minimal
    # list would permanently misroute native providers (e.g. anthropic -> custom)
    # even after Hermes becomes importable; returning it un-cached lets the next
    # call retry the import.
    if registry_ok:
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
    if detect_global_hermes_model().get("model"):  # `hermes setup` (~/.hermes)
        return True
    return False


def resolve_model(agent_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Effective {provider, model, base_url, api_key, source, display_provider}.

    Per-agent override (layered over) → swarm default → ~/.hermes. The routing
    provider is the native Hermes provider id, EXCEPT OpenAI-compatible providers
    (OpenRouter/Custom/etc.) which route via ``custom`` + base_url. A custom /
    OpenAI-compatible endpoint is configured the same way as any provider —
    through `hermes setup` (or the dashboard) — not via swarm-specific env vars.
    """
    agent_cfg = agent_cfg or {}
    default = get_default_model()
    if default.get("model"):
        base = dict(default)
        base_source = "default"
    else:
        # No swarm default yet: adopt the user's NATIVE Hermes setup (~/.hermes),
        # which is the canonical place to configure a provider — `hermes setup`
        # writes there.
        glob = detect_global_hermes_model()
        if glob.get("model"):
            base = dict(glob)
            base_source = "hermes"
        else:
            base = {"provider": "", "model": "", "base_url": "", "api_key": ""}
            base_source = "unconfigured"

    ov_model = (agent_cfg.get("model") or "").strip()
    ov_provider = (agent_cfg.get("provider") or "").strip()
    ov_base = (agent_cfg.get("base_url") or "").strip()
    ov_key = (agent_cfg.get("api_key") or "").strip()
    overridden = bool(ov_model or ov_provider or ov_base or ov_key)

    provider = ov_provider or base.get("provider") or "custom"
    # Empty when truly unconfigured (no override, no default, no opt-in proxy):
    # the agent build surfaces a clear "run `hermes setup`" error instead of
    # silently dialing a phantom localhost proxy.
    model = ov_model or base.get("model") or ""

    # api_key: prefer an explicit override key. Otherwise inherit the default's
    # key ONLY when the override stays on the SAME provider (and base_url) as the
    # default — otherwise we'd send e.g. the OpenRouter key to Anthropic and 401
    # every turn. An override that switches provider without its own key gets an
    # empty key (and a warning), not the wrong provider's secret.
    if ov_key:
        api_key = ov_key
    else:
        base_provider = base.get("provider") or ""
        base_base_url = base.get("base_url") or ""
        same_provider = (not ov_provider) or (ov_provider == base_provider)
        same_base_url = (not ov_base) or (ov_base == base_base_url)
        if same_provider and same_base_url:
            api_key = base.get("api_key") or ""
        else:
            api_key = ""
            log.warning(
                "Agent override switches to provider=%s (base_url=%s) with no "
                "api_key configured; not borrowing the default %s key.",
                ov_provider or provider, ov_base or "(native)",
                base_provider or "(none)",
            )

    preset = _preset(provider)
    openai_compat = preset.get("openai_compatible", True) if preset else True

    if openai_compat:
        # Custom path: needs a base_url — inherit/override, else the preset's.
        base_url = ov_base or base.get("base_url") or preset.get("base_url") or ""
        route_provider = "custom"
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


# ---------------------------------------------------------------------------
# Swarm-side pricing fallback for custom / OpenAI-compatible endpoint models
# only — USD per 1M tokens: (input, output, cached_input). When a model is served
# behind a custom base_url, Hermes can't see the real model to price it
# (resolve_billing_route → billing_mode "unknown"), so the /teams/{id}/costs
# endpoint prices those token deltas with this map. NATIVE providers are priced
# by Hermes instead (see estimate_cost_usd), so they do NOT belong here.
# cached_input None = the endpoint reports no cached tier. Extend this map for
# whatever models your custom endpoint serves; unknown models fall back to the
# per-team token budget instead of a dollar one.
# ---------------------------------------------------------------------------
MODEL_PRICES_PER_MILLION: Dict[str, tuple] = {
    "deepseek-v4-flash":  (0.19, 0.51, None),
    "kimi":               (0.19, 0.51, None),
    "gpt-5.4-mini":       (0.25, 2.00, 0.025),
    "gpt-5.4-nano":       (0.05, 0.40, 0.005),
}


def _hermes_estimate_cost_usd(model, input_tokens, output_tokens,
                              cache_read_tokens, cache_write_tokens,
                              provider, base_url) -> Optional[float]:
    """Defer pricing to Hermes' usage_pricing engine (precise per-model rates,
    incl. cache tiers, kept fresh as Hermes ships models). None on any miss."""
    try:
        from agent.usage_pricing import estimate_usage_cost, CanonicalUsage
        result = estimate_usage_cost(
            model,
            CanonicalUsage(
                input_tokens=int(input_tokens), output_tokens=int(output_tokens),
                cache_read_tokens=int(cache_read_tokens),
                cache_write_tokens=int(cache_write_tokens),
            ),
            provider=provider, base_url=base_url or None,
        )
        amt = getattr(result, "amount_usd", None)
        return float(amt) if amt is not None else None
    except Exception:
        return None


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int,
                      cache_read_tokens: int = 0, *, provider: Optional[str] = None,
                      base_url: Optional[str] = None,
                      cache_write_tokens: int = 0) -> Optional[float]:
    """Price a token bundle in USD, or None for unknown models (the dashboard
    shows 'n/a' rather than a wrong number). input_tokens must EXCLUDE cached
    reads (Hermes' canonical usage already subtracts them).

    Custom / OpenAI-compatible endpoint models are opaque to Hermes — it sees
    only the alias served behind the base_url — so they're priced from the swarm
    table below. For a NATIVE provider (the `hermes setup` path) Hermes prices
    precisely, so we DEFER to it: that's how a native-provider user sees correct
    costs without the swarm re-encoding every provider's price sheet (and without
    the table going stale as new models ship)."""
    prices = MODEL_PRICES_PER_MILLION.get((model or "").strip().lower())
    if prices:
        p_in, p_out, p_cache = prices
        if p_cache is None:
            p_cache = p_in * 0.1
        return (input_tokens * p_in + output_tokens * p_out
                + cache_read_tokens * p_cache) / 1_000_000.0
    # Not a proxy-table model → defer to Hermes. Needs a real provider to resolve
    # a billing route; "custom"/proxy and genuinely unknown models come back None.
    if provider and (provider or "").lower() != "custom":
        return _hermes_estimate_cost_usd(
            model, input_tokens, output_tokens, cache_read_tokens,
            cache_write_tokens, provider, base_url)
    return None
