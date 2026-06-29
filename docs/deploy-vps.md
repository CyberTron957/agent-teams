# Deploying Agent Teams on a VPS (or any always-on host)

Agent Teams is built to run **24/7, unattended**. This guide covers a hardened
self-hosted setup. Read the threat model first — it determines everything else.

## Threat model (read this)

Each agent is a full Hermes agent: it can **run terminal commands, read/write
files, and drive a browser** — all as the OS user the server runs as. That power
is the point (agents ship real work), but it means:

- **Anyone who can reach the API can make agents act.** There is no per-user
  permission model — it's a single trust boundary. Protect it.
- **An agent that goes off the rails runs with your server user's privileges.**
  Contain the blast radius (dedicated user, or Docker).

Two consequences drive the rest of this doc:

1. **Never expose the port without `SWARM_API_KEY`.** With it set, every HTTP
   endpoint *and* the WebSocket require the key; the dashboard prompts for it
   once and stores it in your browser.
2. **Prefer containment.** Docker (below) is the easiest way to keep agent
   terminal access off your host filesystem. On bare metal, use the dedicated
   user + the systemd hardening in `deploy/agent-teams.service`.

---

## Option A — Docker behind a TLS reverse proxy (recommended)

The image bundles Python, Hermes, Chromium and the dashboard, and runs the
agents inside the container — so their terminal access is contained.

```bash
git clone <repo> agent-teams && cd agent-teams
cp .env.example .env
# Edit .env:
#   SWARM_API_KEY=<a long random string>     → REQUIRED when exposed
# Then configure the provider with the Hermes wizard (persists on the volume):
#   docker compose run --rm -e HERMES_HOME=/data/.hermes-shared swarm hermes setup
```

By default `docker-compose.yml` binds `127.0.0.1:8000` only. Put a TLS proxy in
front rather than exposing 8000 directly. Caddy is the shortest path:

```caddyfile
# /etc/caddy/Caddyfile
swarm.example.com {
    reverse_proxy 127.0.0.1:8000
}
```

Caddy auto-provisions HTTPS. WebSockets pass through `reverse_proxy` with no
extra config. nginx equivalent (the `Upgrade`/`Connection` headers matter — the
live execution view and the browser handover are WebSockets):

```nginx
server {
    server_name swarm.example.com;
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 3600s;   # long-lived WS / screencast streams
    }
    # ...listen 443 ssl + certbot-managed cert...
}
```

Then `docker compose up -d --build`. Open `https://swarm.example.com`, enter the
API key when prompted.

> **Firewall:** allow 80/443 only. Keep 8000 (and any local LLM endpoint) bound
> to localhost / the Docker network — never the public interface.

---

## Option B — bare metal with systemd

For a host where Docker isn't an option. Contain the agents with a dedicated
user; `deploy/agent-teams.service` adds systemd sandboxing on top.

```bash
sudo useradd --system --create-home --home-dir /var/lib/agent-teams hermes
sudo -u hermes python3 -m venv /var/lib/agent-teams/venv
sudo -u hermes /var/lib/agent-teams/venv/bin/pip install /path/to/agent-teams
sudo -u hermes /var/lib/agent-teams/venv/bin/playwright install chromium

# Configure the provider as the service user (writes ~/.hermes for that user;
# the swarm adopts it). Custom / OpenAI-compatible endpoint → pick "custom".
sudo -u hermes HOME=/var/lib/agent-teams /var/lib/agent-teams/venv/bin/hermes setup

sudo tee /etc/agent-teams.env >/dev/null <<'EOF'
SWARM_API_KEY=<a long random string>
SWARM_DATA_DIR=/var/lib/agent-teams/data
SWARM_LOG_FILE=/var/lib/agent-teams/swarm.log
SWARM_HOST=127.0.0.1
EOF
sudo chmod 600 /etc/agent-teams.env

sudo cp deploy/agent-teams.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now agent-teams
```

Front it with the same Caddy/nginx TLS proxy as Option A. Check it:

```bash
systemd-analyze verify deploy/agent-teams.service   # validate the unit
systemctl status agent-teams
curl -s localhost:8000/health | jq                   # liveness only without the key
journalctl -u agent-teams -f                         # or tail $SWARM_LOG_FILE
```

`Restart=on-failure` brings the server back after a crash; the lifespan
finalizer drains agents and flushes browser cookies on `systemctl stop`.

---

## Operating it 24/7

- **Budgets.** Set a per-team daily spend cap from the dashboard (click the
  cost badge). When a team hits it, its agents pause — work is **held, not
  lost** — and auto-resume at 00:00 UTC, or when you raise the cap / click
  *Resume anyway*. This is the safety net against an overnight runaway bill.
- **Browser logins.** When an agent needs you to log in / clear a CAPTCHA, it
  posts a takeover request to your inbox. Click **Open browser** to drive its
  (headless) session live from the dashboard, then **Done — hand back**. No
  display needed on the server. (`SWARM_TAKEOVER_MODE=window` keeps the old
  pop-a-Chrome-window behaviour for local desktop use.)
- **Health.** `GET /health` returns liveness to anyone and the full picture
  (uptime, queue depth, LLM-backend reachability) to an authenticated caller —
  point your uptime monitor at it.
- **Logs.** stdout always; set `SWARM_LOG_FILE` for an on-disk rotating trail.
- **Data & backups.** All state is under `SWARM_DATA_DIR`; every config save
  keeps a rotating backup in `<data>/config_backups/`. Back up that directory.
- **Secrets.** Team credentials and the LLM key are stored on disk (the
  credentials file is `0600`). Keep `SWARM_DATA_DIR` and `/etc/agent-teams.env`
  off any world-readable path; treat the host as holding live secrets.

---

## Quick checklist before you expose it

- [ ] `SWARM_API_KEY` set to a long random value
- [ ] TLS reverse proxy in front; only 80/443 open to the world
- [ ] 8000 (and any local LLM endpoint) bound to localhost / private network
- [ ] Running in Docker, or as a dedicated non-root user via systemd
- [ ] A per-team daily budget set
- [ ] `SWARM_DATA_DIR` backed up; secrets files `chmod 600`
