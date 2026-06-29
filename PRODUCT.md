# Product

## Register

product

## Users

Self-hosters and AI tinkerers running a 24/7 multi-agent swarm on their own
hardware. They are technical (comfortable with Docker, API keys, terminals)
and use the dashboard as an operations console: watching agents think and call
tools in real time, steering the team, approving config changes, and answering
agents' questions from the human inbox. Sessions are long-running and often
glanced at rather than actively driven — the UI must stay legible at a
distance and trustworthy at 3am.

## Product Purpose

Agent Teams is a self-hostable multi-agent swarm server with a real-time
dashboard. Each agent is a full Hermes agent (web browsing, terminal, files,
publishing); agents collaborate peer-to-peer on shared projects. The dashboard
is the single pane of glass for the whole team: live execution view, per-agent
configuration, telemetry, and a human-in-the-loop inbox. Success looks like an
operator who can understand swarm state in seconds and intervene with
confidence.

## Brand Personality

Mission-control calm. Precise, quiet confidence — information-dense but never
frantic. The feel of an observability tool you trust during an incident:
tabular numbers, hairline borders, restrained motion, one electric accent.
Three words: precise, calm, alive.

## Anti-references

- Generic AI-startup slop: purple gradients everywhere, glassmorphism as
  default, hero metrics, identical card grids.
- Enterprise admin bloat: heavy chrome, endless undifferentiated tables,
  Bootstrap-era density without hierarchy.
- Crypto-dashboard neon: glow overload, neon-on-black, decorative charts.
- Grey cards with green sliders — the lazy "settings panel" default.
- Emojis in place of proper icons; every glyph should be a real, consistent
  icon.

## Design Principles

1. **State at a glance** — the operator should read swarm health (who's idle,
   busy, blocked on a human) in under three seconds, from across the room.
2. **Live, not noisy** — real-time data earns subtle motion (pulses, streams),
   never attention-grabbing animation; the UI breathes, it doesn't flash.
3. **Density with hierarchy** — pack information in, but every panel has one
   clear primary signal; secondary detail recedes via the muted/dim text ramp.
4. **Trust through precision** — tabular numerals, exact timestamps, honest
   status colors; the dashboard never decorates data.
5. **Operator in control** — every agent action is inspectable and every
   intervention (config, inbox reply, approval) is one obvious click away.

## Accessibility & Inclusion

Target WCAG AA: ≥4.5:1 text contrast on the dark theme (watch the dim-text
ramp), full keyboard operability for inbox/config flows, visible focus
indicators, and `prefers-reduced-motion` alternatives for all live-status
animation (pulses, streaming indicators).
