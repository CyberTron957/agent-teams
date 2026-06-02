# Hermes Swarm — self-contained image (Python + Hermes + Chromium + dashboard).
FROM python:3.12-slim

# Minimal system deps (git for any VCS deps; certs/curl for healthchecks).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Install the swarm + its deps (pulls hermes-agent[all]).
RUN pip install --no-cache-dir .

# Chromium for the browser-publishing tools (+ its OS libraries). Falls back to
# installing the playwright CLI first if hermes-agent[all] didn't provide it.
RUN python -m playwright install --with-deps chromium \
    || (pip install --no-cache-dir playwright \
        && python -m playwright install --with-deps chromium)

# Persistent writable state (configs, queues, agent workspaces, monitoring db).
ENV SWARM_DATA_DIR=/data \
    SWARM_HOST=0.0.0.0 \
    SWARM_PORT=8000
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["hermes-swarm", "up"]
