<h1 align="center">Deploy autonomous AI teams that grow your business while you sleep.</h1>

<p align="center">
  Agent Teams gives you a workforce of AI specialists that
  collaborate, execute, and deliver.
  <br/><br/>
  <strong>Self-hosted. Open source. Industry-agnostic.</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License" /></a>
  <a href="https://github.com/CyberTron957/agent-teams/stargazers"><img src="https://img.shields.io/github/stars/CyberTron957/agent-teams?style=flat" alt="Stars" /></a>
  <a href="https://discord.gg/bcg7HkzPx"><img src="https://img.shields.io/badge/discord-join-5865F2?logo=discord&logoColor=white" alt="Discord" /></a>
</p>

<p align="center">
  <img src="assets/dashboard.png" alt="Agent Teams dashboard — agent roster, supervisor graph, live execution, and cost tracking" width="720" />
</p>

<p align="center">
  <a href="#quickstart"><strong>Quickstart</strong></a> &middot;
  <a href="docs/getting-started.md"><strong>Docs</strong></a> &middot;
  <a href="docs/deploy-vps.md"><strong>Deploy</strong></a> &middot;
  <a href="#license"><strong>License</strong></a>
</p>

<br/>

---

<br/>

## What did my AI team do overnight?

```text
✓ Researched 14 competitor landing pages and wrote a positioning memo
✓ Published 3 social media posts and scheduled 5 more
✓ Found and contacted 22 inbound leads with personalized outreach
✓ Built a lead scoring dashboard from CRM data
✓ Analyzed last week's campaign performance and proposed optimizations
✓ Asked me two questions

I woke up to a summary.
```

Agent Teams runs 24/7 on your machine or VPS. When you wake up, your agents have been working all night. No babysitting. No context loss. No runaway costs.

<br/>

---

## Why this exists

[**Hermes**](https://github.com/NousResearch/hermes-agent) gives you one AI employee — a powerful agent with a terminal, browser, and file access.

**Agent Teams gives you an entire AI company.**

| | One agent | A team of agents |
|---|---|---|
| **Work** | One task at a time | Multiple specialists collaborating |
| **Coverage** | Only when you're watching | 24/7 autonomous execution |
| **Oversight** | You check the output | Supervisors review, nudges correct |
| **Recovery** | Restart from scratch | In-flight tasks survive reboots |
| **Costs** | You notice the bill later | Daily budgets auto-pause spend |

Hermes is the worker. Agent Teams is the manager, the scheduler, the budget controller, and the operations console.

<br/>

<div align="center">
  <table>
    <tr>
      <td align="center"><img src="assets/logos/openai.svg" height="28" alt="OpenAI" /><br/><sub>OpenAI</sub></td>
      <td align="center"><img src="assets/logos/claude-color.svg" height="28" alt="Anthropic" /><br/><sub>Anthropic</sub></td>
      <td align="center"><img src="assets/logos/gemini-color.svg" height="28" alt="Google Gemini" /><br/><sub>Gemini</sub></td>
      <td align="center"><img src="assets/logos/ollama.svg" height="28" alt="Ollama" /><br/><sub>Ollama</sub></td>
      <td align="center"><img src="assets/logos/qwen-color.svg" height="28" alt="Qwen" /><br/><sub>Qwen</sub></td>
      <td align="center"><img src="assets/logos/deepseek-color.svg" height="28" alt="DeepSeek" /><br/><sub>DeepSeek</sub></td>
      <td align="center"><img src="assets/logos/kimi-color.svg" height="28" alt="Kimi" /><br/><sub>Kimi</sub></td>
      <td align="center"><img src="assets/logos/openrouter.svg" height="28" alt="OpenRouter" /><br/><sub>OpenRouter</sub></td>
      <td align="center"><img src="assets/logos/zai.svg" height="28" alt="Z.ai" /><br/><sub>Z.ai</sub></td>
    </tr>
  </table>
  <br/>
  <em>40+ providers via Hermes. Or route everything through one OpenAI-compatible endpoint.</em>
</div>

<br/>

---

<br/>

## Problems Agent Teams solves

| Without Agent Teams | With Agent Teams |
| ------------------- | ---------------- |
| ❌ You run agents in separate terminals and lose track of what each one is doing. | ✅ One dashboard shows every agent's state, live execution stream, message history, and cost in real time. |
| ❌ An agent gets stuck or starts looping — burning tokens until you notice. | ✅ Self-correction detects loops and stalls, supervisor agents nudge drift, and daily budgets auto-pause runaway spend. |
| ❌ You need to hand a credential or answer a question but the agent already ended its turn. | ✅ Human inbox — agents ask, you answer, they resume. Browser takeover for CAPTCHAs and 2FA. |
| ❌ Context grows unbounded and costs spiral as sessions get longer. | ✅ Automatic context compaction, tool-result aging, stale-payload stubbing — context stays bounded. |
| ❌ A server restart loses all in-flight agent work. | ✅ SQLite task queues recover processing tasks on restart. Session rotation persists compaction across reboots. |

<br/>

---

## How Agent Teams compares

| | Claude Code | Hermes | Agent Teams |
|---|---|---|---|
| **Persistent multi-agent teams** | ⚠️ Experimental | ⚠️ Kanban board | ✅ |
| **Supervisor agents** | ❌ | ❌ | ✅ Dedicated overseers |
| **Self-correction (loop/stall detection)** | ❌ | ❌ | ✅ Automatic |
| **Cost controls with hard stops** | ❌ | ❌ | ✅ Per-team daily budgets |
| **Human-in-the-loop inbox** | ❌ | ⚠️ Telegram gate | ✅ Structured inbox |
| **AI team builder** | ❌ | ❌ | ✅ The Architect |
| **24/7 autonomous execution** | ❌ | ⚠️ Single-agent | ✅ Multi-agent teams |
| **Browser takeover (auth/CAPTCHA)** | ❌ | ❌ | ✅ CDP screencast |

<br/>

---

<br/>

## Example teams you can build

**Marketing Team**

```text
Strategist    → Researches trends, identifies opportunities, defines positioning
Copywriter    → Writes social posts, emails, blog drafts, ad copy
Analyst       → Tracks campaign performance, segments audiences, reports ROAS
Publisher     → Schedules posts, manages distribution, monitors engagement
```

**Sales Team**

```text
Prospector    → Generates leads, researches accounts, scores opportunities
Outreach      → Personalizes emails, follows up, books meetings
Deal Manager  → Tracks pipeline, manages proposals, coordinates approvals
Analyst       → Monitors quotas, forecasts, identifies churn risks
```

**Support Team**

```text
Triage        → Categorizes tickets, routes to specialists, sets priority
Resolver      → Troubleshoots issues, applies fixes, escalates when stuck
Documenter    → Writes knowledge base articles from resolved tickets
Analyst       → Tracks SLAs, identifies trends, suggests process improvements
```

**Operations Team**

```text
Orchestrator  → Monitors workflows, alerts on failures, restarts stalled jobs
Reporter      → Generates daily summaries, surfaces anomalies, delivers briefs
Compliance    → Checks policy adherence, flags risks, prepares audit logs
```

**Build any of these in minutes with the Architect — the built-in AI team builder that interviews you, designs the team, and builds it on approval.**

<br/>

---

<br/>


---

## Features

<table>
<tr>
<td align="center" width="33%">
<h3>🤖 Autonomous 24/7 Operation</h3>
Agents self-wake, collaborate, and recover from failures — without you watching.
</td>
<td align="center" width="33%">
<h3>🔄 Self-Correction</h3>
Loop detectors, stall alerts, and context compaction prevent runaway spend.
</td>
<td align="center" width="33%">
<h3>👤 Human-in-the-Loop</h3>
Agents ask for decisions. You answer, they resume. CAPTCHA and 2FA supported.
</td>
</tr>
<tr>
<td align="center">
<h3>💰 Cost Controls</h3>
Per-team daily budgets auto-pause agents when limits are hit.
</td>
<td align="center">
<h3>📊 Live Dashboard</h3>
Real-time agent telemetry, network graph, cost badges, and activity digests.
</td>
<td align="center">
<h3>🏗️ The Architect</h3>
AI team builder that interviews you and designs your perfect team.
</td>
</tr>
<tr>
<td align="center">
<h3>🔐 Safety & Security</h3>
Single API key guards all endpoints. Per-team credentials with purpose validation.
</td>
<td align="center">
<h3>🔌 40+ LLM Providers</h3>
OpenAI, Anthropic, Gemini, Ollama, OpenRouter, DeepSeek, and more.
</td>
<td align="center">
<h3>⏰ Cron Scheduling</h3>
Schedule recurring tasks. Agents wake, work, and report automatically.
</td>
</tr>
</table>

<br/>

---

<br/>

## What Agent Teams is not

| | |
|---|---|
| **Not a chatbot.** | Agents have jobs, roles, and teammates — not chat windows. |
| **Not an agent framework.** | We don't tell you how to build agents. Hermes is the agent; Agent Teams is the orchestration layer. |
| **Not a single-agent tool.** | If you have one agent running one task, you probably don't need Agent Teams. If you have five agents collaborating — you do. |
| **Not cloud-hosted.** | Self-hosted on your hardware. Your API keys, your data, your infrastructure. |
| **Not a replacement for Hermes.** | Agent Teams extends Hermes. Every agent IS a Hermes agent. |

<br/>

---

## Quickstart

> **Have an AI coding agent?** Paste one of these prompts into **Claude Code,
> Codex, opencode, or Hermes itself** and it'll install and set up the whole teams
> for you — clone, dependencies, provider, and first run.

<details>
<summary><b>📋 Local install prompt</b> — set it up on this machine</summary>

```text
Install and run Agent Teams on this machine for me.

Agent Teams (https://github.com/CyberTron957/agent-teams) is a
self-hosted multi-agent server with a real-time dashboard.

1. Check Python 3.11+ (or Docker). If missing, tell me before installing system packages.
2. If ./agent-teams doesn't exist, clone the repo there. Run the installer
   non-interactively: `bash install.sh --no-run`. It auto-skips the interactive
   wizard when there's no TTY, so it won't hang. (Add `--no-browser` only if the
   Chromium download fails — everything else still works.)
3. Check whether a provider is already configured — run `.venv/bin/agent-teams doctor`.
   If it already shows a model (e.g. an existing `~/.hermes` setup), ADOPT IT —
   don't ask me for keys, the teams reuses it automatically. Skip to step 5.
4. Only if NO provider is configured, ask me for:
      - provider (e.g. openai, anthropic, or "custom" for an OpenAI-compatible / proxy endpoint)
      - model name, API key, and base URL (base URL only for custom/proxy, e.g. http://localhost:4000/v1)
   Then set it with the supported NON-interactive command (do NOT edit Python internals):
     .venv/bin/agent-teams set-model --provider <p> --model <m> [--base-url <url>] --api-key <key>
   Then tell me that `set-model` only sets the model — for web-search, vision,
   browser providers, memory, and reasoning-effort customization I can run the
    full wizard myself anytime: `.venv/bin/agent-teams setup`. Offer to run it.
5. Verify: `.venv/bin/agent-teams doctor` (a "backend not reachable" warning is
   fine if my model server isn't running yet).
6. Scaffold a team and start the server detached:
     .venv/bin/agent-teams init
      .venv/bin/agent-teams up --detach   # daemonizes and returns; don't background it yourself
   It prints the URL once /health is up. Confirm with `.venv/bin/agent-teams status`.
   (Use `up --detach`, NOT `nohup … &` — backgrounding it from your shell means it
   dies when your session/process group ends.)
7. Tell me how to manage it and build my first team:
      - open the dashboard at http://127.0.0.1:8000 and use the Architect
      - check status:  .venv/bin/agent-teams status
      - stop it:       .venv/bin/agent-teams down
      - customize more: .venv/bin/agent-teams setup

Keep my API key local — don't commit it or send it anywhere.
```

</details>

<details>
<summary><b>📋 VPS install prompt</b> — deploy it on a server, exposed safely</summary>

```text
Deploy Agent Teams on this VPS, exposed safely over HTTPS.

Agent Teams (https://github.com/CyberTron957/agent-teams) is a
self-hosted multi-agent server. Its agents can run terminal commands as the
server user, so containment and auth matter. Please:

1. Read docs/deploy-vps.md in the repo and follow its hardened path. Prefer the
   Docker route so the agents' terminal access stays contained.
2. Clone the repo and bring it up with Docker Compose (restart: unless-stopped).
3. Generate a strong TEAMS_API_KEY and set it — it must guard every endpoint and
   the WebSocket. Show it to me once and store it somewhere I can find it.
4. Configure the LLM provider via `hermes setup` against the shared config —
   PAUSE and ask me for the provider + API key.
5. Put it behind a TLS reverse proxy (Caddy or nginx) for my domain — ask me for
   the domain/subdomain — with automatic HTTPS.
6. Lock down the firewall: only 80/443 and SSH open; do NOT expose the raw app
   port.
7. Verify: GET /health returns ok, and the dashboard loads over HTTPS and prompts
   for the API key.
8. Report back the URL, where the API key is stored, and how to watch logs
   (docker logs / journald).

Ask me before anything destructive. Never print my API key or provider key into a
file that could be committed.
```

</details>


**Or do it yourself — macOS & Linux, one line.**

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/CyberTron957/agent-teams/main/install.sh)
```


<details>
<summary><b>Install with Docker</b> <b> or from a clone</b> </summary>

```bash
# Docker
git clone https://github.com/CyberTron957/agent-teams agent-teams && cd agent-teams
docker compose run --rm teams agent-teams setup  
docker compose up --build
```
or
```bash
# From a clone 
git clone https://github.com/CyberTron957/agent-teams agent-teams && cd agent-teams
bash install.sh
```

</details>

> **Requirements:** Python 3.11+, Chromium (for browser tools: `npx playwright install chromium`)

### After install

```bash
agent-teams setup       # configure provider + model
agent-teams doctor       # preflight check
agent-teams init         # scaffold a starter team
agent-teams up --detach  # start the server
open http://127.0.0.1:8000  # open the dashboard
```

<br/>

---

## CLI Reference

| Command | Description |
| ------- | ----------- |
| `agent-teams up` | Start the server (foreground) |
| `agent-teams up --detach` | Start as a daemonized background process |
| `agent-teams down` | Stop a running server |
| `agent-teams status` | Check if the server is running |
| `agent-teams setup` | Interactive provider/tool wizard (40+ providers) |
| `agent-teams set-model` | Set provider and model non-interactively |
| `agent-teams init` | Scaffold a starter team |
| `agent-teams doctor` | Preflight: checks Hermes, model backend, Chromium |
| `agent-teams update` | Git pull + reinstall to the latest version |

<br/>

---

## FAQ

**How is this different from running multiple Hermes agents in separate terminals?**
Agent Teams gives them a shared project directory, peer-to-peer messaging (they can delegate and report to each other), supervisor agents that watch for stalls, a human inbox for questions, budget enforcement, and one dashboard to see it all. Separate terminals give you none of that.

**Can I use any LLM provider?**
Yes. Hermes supports 40+ providers natively (OpenAI, Anthropic, OpenRouter, Google, Groq, Together, DeepSeek, and more). Or route everything through one OpenAI-compatible endpoint (LiteLLM, custom proxy, local Ollama).

**What happens when an agent's context gets too long?**
Hermes' built-in ContextCompressor auto-compacts sessions at 75% of the context window. Turn summaries and the recent tail are preserved in a child session; the raw history is pruned. Agent Teams also ages stale tool results and sweep payloads at replay.

**Can I stop an agent mid-turn?**
Yes. Pause freezes the queue and preserves pending tasks. Stop interrupts the in-flight turn via Hermes' interrupt() API, drains pending tasks, and waits for the worker thread to unwind.

**Does this work on a headless VPS?**
Yes. The dashboard is served over HTTP, and the embedded browser takeover streams the headless Chrome via CDP screencast — no display needed. The Docker route also contains the agents' terminal access.

**How do costs work?**
Every turn logs token counts (input, output, cache read). These are priced against a built-in provider pricing table and summed per-agent and per-team. Set a daily USD (or token) cap per team — agents auto-pause when the limit is hit and resume at 00:00 UTC.

<br/>

---

## Deploy on a VPS

See **[Deploy on a VPS](docs/deploy-vps.md)** for the hardened deployment guide — TLS reverse proxy, firewall lockdown, Docker containment, and TEAMS_API_KEY setup.

<br/>

---

## Development

```bash
git clone https://github.com/CyberTron957/agent-teams.git
cd agent-teams
python -m venv .venv && source .venv/bin/activate
pip install -e .
agent-teams setup    # configure provider + model
agent-teams init     # scaffold a team
agent-teams up       # start the server
```

<br/>

---

## Community

- [Discord](https://discord.gg/bcg7HkzPx) — chat and community
- [GitHub Issues](https://github.com/CyberTron957/agent-teams/issues) — bugs and feature requests
- [GitHub](https://github.com/CyberTron957/agent-teams) — source code and releases

<br/>

---

## License

MIT &copy; 2026 Pradhyun

<br/>

---

<p align="center">
  <sub>Open source under MIT. Built for people who want AI agents that ship, not just chat.</sub>
</p>
