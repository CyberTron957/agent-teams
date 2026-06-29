# Getting Started with Agent Teams

A complete walkthrough — from downloading the project to running a multi-agent
team that works on its own and asks you for help when it needs it.

> **What this is.** Agent Teams runs a team of AI agents that collaborate on a
> shared project 24/7. Each agent can browse the web, run a terminal, write
> files, and publish to real platforms. You watch and steer the whole team from
> one web dashboard. This guide assumes you're comfortable with a terminal, API
> keys, and (optionally) Docker.

---

## 1. What you need first

1. **An LLM provider API key.** You'll configure it with **`hermes setup`** —
   Hermes' built-in wizard. It supports 40+ providers (Anthropic, OpenAI,
   [OpenRouter](https://openrouter.ai), Groq, DeepSeek, local models, …): pick
   one, paste your key, choose a model. The swarm reads that config directly, so
   there's nothing provider-specific to wire up separately. *(Using a
   [LiteLLM](https://github.com/BerriAI/litellm) proxy or any OpenAI-compatible
   endpoint? It's just another provider — pick **custom** in `hermes setup` and
   enter its base URL + key.)*
2. **Docker** *(recommended)* **or Python 3.11+**.

You do **not** need to install Hermes (the agent runtime) separately — it's
pulled in automatically, and the `hermes` CLI comes with it.

---

## 2. Download & install

Each path ends with the dashboard on **http://127.0.0.1:8000**.

### One line — macOS & Linux

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/CyberTron957/hermes-mission-control/main/install.sh)
```

That clones the repo, sets up a local `.venv`, installs everything, fetches a
browser, runs `hermes setup` if you don't already have a provider (40+ providers,
incl. a custom / OpenAI-compatible endpoint), scaffolds a starter team, verifies
with `agent-teams doctor`, and offers to start the dashboard. **Already have
`~/.hermes`?** It adopts your provider + keys and skips setup. Safe to re-run.
*(Flags: `--no-run`, `--no-setup`, `--no-browser`, `--yes`.)*

**Windows:** run that line in [WSL](https://learn.microsoft.com/windows/wsl/), or
use Docker below. **Already cloned the repo?** `bash install.sh` does the same.

### Docker

Runs the agents *inside* the container (keeping their terminal access off your
host); data persists in the `swarm-data` volume.

```bash
git clone <this-repo> agent-teams && cd agent-teams
docker compose run --rm -e HERMES_HOME=/data/.hermes-shared swarm hermes setup
docker compose up --build
```

The middle step picks your provider with Hermes' wizard (written to the shared
config on the volume, so it persists). Using a LiteLLM proxy or other
OpenAI-compatible endpoint? Choose **custom** and enter its base URL + key (for
one on your host, use `http://host.docker.internal:<port>`).

> `agent-teams doctor` is the fastest way to diagnose a bad install — it checks
> that Hermes imports, a model is configured (and reachable), Chromium is
> available, and the Hermes compat seams hold, telling you exactly what to fix.

---

## 3. First run

Open the dashboard. The **default model** (the one new agents use unless you
override per-agent) is whatever you picked in `hermes setup`. You can change it
anytime from the dashboard — it reads the available models live and writes your
choice back to the same Hermes config.

You now have an empty swarm. There are two ways to build your first team — let
the **Architect** do it for you (recommended), or wire it by hand.

---

## 4. Build a team — the easy way (the Architect)

Click **Architect** in the top bar. The Architect is an AI assistant that knows
the whole framework. It is *not* part of any team — it's your team builder.

Tell it what you want to accomplish, e.g.:

> "I want a small team that researches AI news daily and drafts a LinkedIn post
> for me to approve."

It will:

1. **Ask a few focused questions** about your goal and constraints.
2. **Propose a team** — the agents, each agent's role ("soul"), who talks to
   whom, and a shared brief — for you to review in chat.
3. **Build it live once you approve** — create the agents, write their souls,
   link them, seed the shared `workspace.md`, and offer to kick them off.

You can also ask it to change an existing team later: *"add a QA agent to
acme"*, *"rewrite the coordinator's brief to be stricter about shipping"*.

> Tip: you can pick the Architect's own model in the **Model** settings, and it
> has web search so it can ground its suggestions.

---

## 5. Build a team — by hand (concepts)

If you'd rather wire things yourself (or just want to understand what the
Architect builds), here are the pieces. Everything below is editable live from
the dashboard's ⚙️ panel — no restart.

- **Team** — a group of agents sharing one project directory and one shared
  brief. Create one with **+ New Team**.
- **Agent** — one Hermes worker. Its key settings:
  - **Role / soul** — a short charter (≈100–200 words) telling the agent who it
    is and what it owns. Crisp, non-overlapping mandates work best.
  - **Connections (peers)** — who this agent may message. **Connections are
    bidirectional**: linking A to B lets both message each other. Agents can
    only talk to peers on the same team.
  - **Supervisor** — a supervisor periodically reviews its linked teammates'
    work and nudges them if they stall (good for a coordinator/manager role).
  - **Autonomous** — an autonomous agent wakes itself on an interval to push the
    mission forward without waiting for a task. Keep **one** autonomous "driver"
    per team (usually the coordinator) so the team has momentum without
    everyone self-triggering at once.
  - **Model** — defaults to the swarm default; override per agent if you want a
    cheaper model for grunt work and a stronger one for the lead.
- **Shared workspace** — `workspace.md` is a shared brief injected into every
  agent's context; the team's `project/` directory is their shared working area.

You give an agent work by sending it a **task** from its card. From there it
runs, calls tools, talks to peers, and reports back — all visible live.

---

## 6. Watch & steer the team

The dashboard is your mission control:

- **Live execution view** — watch each agent think → call a tool → answer, in
  real time.
- **Network view** — see the agents and the connections between them.
- **Per-agent telemetry & config** — open an agent to see its state and tune it.
- **Self-aware agents** — agents can read their own config and *propose* changes
  (e.g. "raise my iteration limit"); you approve or reject from the inbox.

### The human inbox

Agents ask you for things — a decision, a credential, or a login they can't do
themselves. These land in the **Inbox** (top bar). Two kinds:

- **Questions** — click **Respond** and type your answer; the agent resumes with
  it.
- **Browser takeovers** — when an agent hits a login / CAPTCHA / 2FA, it asks
  you to take over its browser. Click **Open browser** to drive the agent's live
  (headless) browser session right inside the dashboard — click, type, switch
  tabs, navigate — then click **Done — hand back**. The agent resumes on the
  now-authenticated session. This works even on a display-less server. (You can
  also click **Browser** in the top bar any time to watch a team's browser.)

### Credentials

Instead of pasting a password into chat, store it once per team
(`/teams/{team}/credentials`, or via the UI). Agents reference it by name and
use it through the browser; the secret itself is stored on disk with `0600`
permissions and is never echoed back. For interactive logins, use the browser
takeover above rather than handing over a password.

---

## 7. Keep spending under control (budgets)

A 24/7 swarm on a paid API can run up a bill overnight. Set a **per-team daily
budget**: click the cost badge in the top bar and enter a USD cap.

When a team reaches its cap, its agents **pause** — in-flight work is **held,
not lost** — and a banner appears. The team auto-resumes at **00:00 UTC**, or
immediately when you **Raise limit** or click **Resume anyway**. Leave it at 0
for unlimited. (For models with no known price, set a token cap instead — the UI
warns you when that applies.)

---

## 8. Schedule recurring work (cron)

Every agent can run work on a schedule — a 9am competitor check, an hourly
metrics pull, a Monday digest. Add/enable/delete schedules from the agent's ⚙️
panel using:

- 5-field cron (`0 9 * * *`),
- shortcuts (`@hourly`, `@daily`, `@weekly`, `@monthly`),
- or intervals (`@every 30m`).

Agents can also schedule and cancel their own wake-ups via the
`schedule_wakeup` / `cancel_wakeup` tools.

---

## 9. Going beyond your laptop (exposing it safely)

By default the server binds `127.0.0.1` with no authentication — fine for local
use. **The moment you expose the port (VPS/LAN), set `SWARM_API_KEY`.** With it
set, every endpoint *and* the live WebSocket require the key, and the dashboard
prompts for it once (then remembers it in your browser).

Also put it behind a TLS reverse proxy, and prefer the Docker route so agents'
terminal access stays contained. Full hardened setup — Docker + Caddy/nginx, or
bare-metal with the included systemd unit, plus a threat model — is in
**[`docs/deploy-vps.md`](deploy-vps.md)**.

---

## 10. Data, backups, and logs

- **State** lives under `SWARM_DATA_DIR` (the `swarm-data` volume in Docker, or
  `./data` / `~/.agent-teams/data` otherwise): team configs, task queues,
  per-agent conversation history, the shared project, and credentials.
- **Backups**: every config save keeps a rotating copy under
  `<data>/config_backups/`. Back up the whole `SWARM_DATA_DIR` to be safe.
- **Logs**: always on stdout (so `docker logs` / journald capture them). Set
  `SWARM_LOG_FILE=/path/to/swarm.log` for an on-disk rotating trail too.
- **Health**: `GET /health` reports liveness to anyone and the full picture
  (uptime, queue depth, LLM-backend reachability) to an authenticated caller —
  point an uptime monitor at it.

---

## 11. Restarts are safe

Stop and restart the server freely — in-flight tasks are recovered and resumed,
conversation history persists, and a team that was over budget when you stopped
stays correctly paused (the meter is rebuilt from history on startup). Browser
logins persist across restarts too.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Dashboard loads but nothing works / 401s | `SWARM_API_KEY` is set — enter it when prompted (or unset it for local use). |
| "Hermes NOT importable" | `pip install hermes-agent`, or set `HERMES_AGENT_PATH`. Run `agent-teams doctor`. |
| "LLM backend NOT reachable" | Re-run `hermes setup` (or `agent-teams doctor`) — check the provider key, and for a custom endpoint that its base URL serves the chosen model. |
| Browser tools unavailable | `playwright install chromium` (or install Chrome). Everything else still works. |
| Agent/Architect can't read a page or describes a URL wrongly | `web_extract` had no backend. Search works out of the box; for extract either configure a Hermes web backend (Firecrawl/Tavily/Exa) or install `pip install .[web]` (crawl4ai) for JS-heavy sites. A configured Hermes backend is always used as-is. |
| Agents idle and doing nothing | Send a task, or mark the coordinator **autonomous** so it self-drives. |
| Costs climbing fast | Set a per-team daily budget (§7). |

Run `agent-teams doctor` whenever something's off — it pinpoints which of the
three prerequisites (Hermes, LLM backend, Chromium) is the problem.
