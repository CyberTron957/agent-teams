# Agent Teams — self-contained image (Python + Hermes + Chromium + dashboard).
# Uses Debian Bookworm (stable) for reliable package repos.
FROM python:3.12-slim-bookworm

# System deps: git for VCS, curl for healthchecks, Chromium deps for browser tools.
# Only use the main bookworm repo (skip -updates/-security which may have future
# timestamps on some mirrors, causing "not valid yet" errors during Docker builds).
RUN echo "deb http://deb.debian.org/debian bookworm main" > /etc/apt/sources.list.d/bookworm-main.list \
    && rm -f /etc/apt/sources.list.d/debian.sources /etc/apt/sources.list \
    && apt-get update -o Acquire::Check-Valid-Until=false \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl git \
        # Chromium system dependencies
        libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 \
        libcups2 libdrm2 libdbus-1-3 libxkbcommon0 libatspi2.0-0 \
        libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 \
        libpango-1.0-0 libcairo2 libasound2 libwayland-client0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Install the swarm + its deps (pulls hermes-agent[all]).
RUN pip install --no-cache-dir .

# Chromium for the browser-publishing tools.
# Try Playwright's bundled Chromium with pre-installed deps first.
# If that fails, try system chromium package. If all fails, warn but continue.
RUN python -m playwright install --with-deps chromium 2>/dev/null \
    || (echo "deb http://deb.debian.org/debian bookworm main" > /etc/apt/sources.list.d/bookworm-main.list \
        && rm -f /etc/apt/sources.list.d/debian.sources /etc/apt/sources.list \
        && apt-get update -o Acquire::Check-Valid-Until=false \
        && apt-get install -y --no-install-recommends chromium 2>/dev/null \
        && rm -rf /var/lib/apt/lists/*) \
    || echo "WARN: Chromium install failed — browser tools will be unavailable"

# Persistent writable state (configs, queues, agent workspaces, monitoring db).
ENV SWARM_DATA_DIR=/data \
    SWARM_HOST=0.0.0.0 \
    SWARM_PORT=8000
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["agent-teams", "up"]
