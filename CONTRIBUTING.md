# Contributing

Thanks for your interest in Agent Teams.

## Dev setup

```bash
git clone https://github.com/CyberTron957/hermes-mission-control agent-teams && cd agent-teams
python3 -m venv .venv && source .venv/bin/activate
pip install -e . pytest
```

`hermes-agent` is pulled in automatically. To run against a local Hermes
checkout instead, set `HERMES_AGENT_PATH=/path/to/hermes-agent`.

## Tests

```bash
pytest tests/ -q
```

Before opening a PR, run the tests and `agent-teams doctor` — `doctor` includes
the Hermes compatibility self-check.

## Pull requests

- Keep each PR focused on one logical change.
- Match the surrounding code's style; no unrelated reformatting.
- Agent Teams is built **over** [Hermes](https://github.com/NousResearch/hermes-agent) —
  prefer deferring to Hermes (models, providers, pricing, config) over
  re-implementing it. The compat self-check guards the seams we do reach into.

## Bugs & ideas

Open an issue with reproduction steps (for bugs) or your use case (for features).
Include your install method and `agent-teams doctor` output for install issues.
