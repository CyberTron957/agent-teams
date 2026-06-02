# Hermes Swarm

A self-hostable **multi-agent swarm server** with a real-time dashboard. Each
agent is a full [Hermes](https://github.com/NousResearch/hermes-agent) agent —
it can browse the web, run a terminal, write files, and publish to live
platforms — and agents collaborate peer-to-peer on a shared project. You watch
and steer the whole team from one web UI.

- **Live execution view** — see each agent think → call tools → answer, in real time.
- **Per-agent config** — model, provider, tools, reasoning, iterations, sampling — from the UI.
- **Self-aware agents** — agents can read their own config/telemetry and *propose* changes for your approval.
- **Human inbox** — agents ask you for logins/decisions; you reply from the dashboard.

---

## Prerequisites

1. **An OpenAI-compatible LLM endpoint + API key** — e.g. [OpenRouter](https://openrouter.ai),
   OpenAI, or your own [LiteLLM](https://github.com/BerriAI/litellm) proxy.
2. **Docker** *(option A)* **or Python 3.11+** *(option B)*.

You do **not** need to install Hermes separately — it comes in automatically
(see [Already have Hermes?](#already-have-hermes) if you do).

---

## Option A — Docker (recommended)

```bash
git clone <this-repo> hermes-swarm && cd hermes-swarm
cp .env.example .env          # then edit .env: set SWARM_LLM_BASE_URL + SWARM_LLM_API_KEY
docker compose up --build
```

Open **http://127.0.0.1:8000**. The image bundles Python, Hermes, Chromium, and
the dashboard; your data persists in the `swarm-data` volume.

> Pointing at an LLM proxy running on your **host** machine? Use
> `SWARM_LLM_BASE_URL=http://host.docker.internal:4000/v1` in `.env`.

---

## Option B — pip

```bash
python3 -m venv .venv && source .venv/bin/activate   # Python 3.11+
pip install .                       # pulls hermes-agent + deps
playwright install chromium         # for the browser-publishing tools

export SWARM_LLM_BASE_URL=https://openrouter.ai/api/v1
export SWARM_LLM_API_KEY=sk-...
export SWARM_DEFAULT_MODEL=openai/gpt-4o-mini

hermes-swarm doctor                 # check Hermes + model backend + Chromium
hermes-swarm init                   # scaffold a starter team + coordinator agent
hermes-swarm up                     # serve the dashboard on http://127.0.0.1:8000
```

---

## Already have Hermes?

If you already use Hermes (`~/.hermes/`), the swarm reuses the same agent
runtime. Resolution order: the pip-installed `hermes-agent`, else
`HERMES_AGENT_PATH`, else `~/.hermes/hermes-agent`. Run `hermes-swarm doctor` to
confirm what it found.

---

## Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `SWARM_LLM_BASE_URL` | `http://127.0.0.1:4000/v1` | OpenAI-compatible LLM endpoint |
| `SWARM_LLM_API_KEY` | `sk-1234` | key for that endpoint |
| `SWARM_DEFAULT_MODEL` | `litellm-model` | default model for new agents |
| `SWARM_FALLBACK_MODELS` | `litellm-model,kimi` | dropdown list if the backend can't be queried |
| `SWARM_HOST` / `SWARM_PORT` | `127.0.0.1` / `8000` | dashboard bind address |
| `SWARM_API_KEY` | *(unset)* | if set, required on mutating endpoints — **set it if you expose the port** |
| `SWARM_DATA_DIR` | repo `./data` or `~/.hermes-swarm/data` | writable state (configs, queues, workspaces) |
| `HERMES_AGENT_PATH` | *(unset)* | path to a Hermes source checkout (only if not pip-installed) |

Per-agent settings (model, provider, tools, reasoning, iterations, sampling,
soul) are edited live from the dashboard's ⚙️ panel — no restart needed.

---

## Notes

- **Security:** the dashboard has no login. Keep it on `127.0.0.1`, or set
  `SWARM_API_KEY` and put it behind a reverse proxy before exposing it.
- **Chromium** is required only for the browser-publishing tools; everything
  else works without it (the swarm degrades gracefully).
- **Data & backups:** state lives in `SWARM_DATA_DIR`; every config save keeps a
  rotating backup under `<data>/config_backups/`.
