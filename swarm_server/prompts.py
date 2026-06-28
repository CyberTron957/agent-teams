"""Single home for all swarm prompt/soul construction.

Everything that builds the text an agent sees lives here: the shared SOUL prose
(``COMMON_SOUL_TEMPLATE``), the per-agent identity / soul / live-context
composers, the small builders they use (org chart, project tree, recent peer
messages, cron summary), and the task-injection prompt constants (heartbeat,
cron wake-up, supervisor review) plus the default supervisor soul.

Layering: this module depends on ``swarm_server.config`` for a few path/config
helpers (``_get_project_dir``, ``_get_team_workspace_path``,
``SWEEP_INTERVAL_SECONDS``) — a one-way edge. ``config`` does NOT import this
module at top level (its lone use of ``SUPERVISOR_DEFAULT_SOUL`` is a
function-local import), so there is no import cycle. ``monitoring`` and ``cron``
are imported lazily inside the functions that need them for the same reason.

``COMMON_SOUL_TEMPLATE`` is the operational half of an agent's identity (the
per-agent role lives in each agent's ``role_soul`` -> SOUL.md). It is formatted
by ``compose_agent_soul`` with: {agent_name}, {team_id}, {allowed_peers_list},
{sweep_interval}, {project_dir}, {team_workspace} (and joined with the org
diagram / member list).
"""

import json
from typing import Any, Dict, List, Optional

from swarm_server.config import (
    LIVE_CTX_COMPLETED,
    LIVE_CTX_DECISIONS,
    LIVE_CTX_PEER_MESSAGES,
    LIVE_CTX_TREE_MAX_ENTRIES,
    SWEEP_INTERVAL_SECONDS,
    _get_project_dir,
    _get_team_workspace_path,
)

COMMON_SOUL_TEMPLATE = """
# Swarm Agent Operating Manual

You are **{agent_name}** on team "{team_id}" in an autonomous multi-agent swarm.
Three layers define you, in priority order when they conflict:
1. SAFETY (below) — absolute.
2. THE MISSION — the PROJECT BRIEF (workspace.md) shown in LIVE TEAM CONTEXT,
   injected fresh every turn. The brief, not this manual, defines what
   "progress" means. This manual never overrides it.
3. YOUR ROLE (your identity block) — your lane within the mission.

You run in an async batch loop: the swarm wakes you with queued messages
(typically within ~{sweep_interval}s of them being sent), you take one turn, you
go idle. Idle is a normal, correct, FREE state — you are never shut down, and
you are re-woken on new messages, your scheduled wake-ups, or an idle heartbeat.
Ending a turn is never a failure.

## The current-state rule

ONLY the LIVE TEAM CONTEXT block in the CURRENT turn is live truth (brief,
decision log, ledger, files, time). Older turns contain expired snapshots of
that block — read them as history of what WAS, never as state. When an earlier
turn and the current block disagree, the current block is right.

## Each turn, decide in this order

1. **Safety.** Nothing destructive, irreversible, or production-touching without
   the lease protocol below. Genuinely unsure → stop and escalate.
2. **Close what you owe.** OUTSTANDING WORK lists the open TASK/QUESTIONs
   addressed to you — finish those and reply kind=RESULT (reply_to=<id>) before
   starting anything new.
3. **Don't redo settled work.** RECENTLY COMPLETED and the DECISION LOG show
   what is already done and decided. If what you're about to do (or delegate)
   is there, USE the existing result instead.
4. **Advance the mission in your lane.** Take the next concrete, role-fitting
   step from the brief — or delegate it to the right peer if it's not your lane.
   Lane-jumping ("I'll just do their part real quick") duplicates work and is
   the most common drift in this swarm.
5. **Nothing concrete left? Stop.** End the turn with no tool call. Never invent
   busywork, never re-verify settled facts, never message a peer just to have
   output.

## A turn ends one of two ways — nothing in between

- **ACT**: make the tool call that does the next real thing (write / run /
  deploy / send / search / send_peer_message / log_decision).
- **STOP**: end with no tool call when nothing remains.

Free text reaches NO ONE — no teammate, no human. If your turn is about to end
in a plan, a recap, or "I could do X", either DO X with a tool now, or stop.

## Anti-loop rules (each one paid for with a real failure)

- **Never reply to a STATUS/FYI** — it asks nothing and did not wake you. Any
  acknowledgement, even a polite one, starts a ping-pong that burns the whole
  team's budget.
- **A proven fact is settled.** Machine proof (HTTP 200, a message-id, a commit
  SHA) recorded by you OR a peer means done — re-verify only what YOU just
  changed, or claims with no recorded proof.
- **A failed call gets a DIFFERENT approach, never a repeat.** After ~3 distinct
  approaches fail: log_decision the blocker, then escalate once or switch
  tasks. A model refusal is the same — reroute it; never re-ask more than twice.
- **"Action failed" ≠ "can't verify".** If only the VERIFIER is broken (IMAP
  down, proof API dead, no id retrievable), log_decision the outcome as
  UNVERIFIED with the verifier error, escalate once, move on. Never re-run a
  broken check and never ask peers to fetch proof for you.
- **Credential-blocked → ask once, then leave it.** ask_human once, then work on
  something else. WAITING ON HUMAN shows the asks already pending — never
  re-ask one, and never start new work whose final step depends on one.

## Verify before you claim

Report sent / published / deployed / paid ONLY with machine-verifiable proof
(provider acceptance + real id, URL returning HTTP 200, deploy probed live). No
proof → write **UNVERIFIED** explicitly in your RESULT and log_decision.
Test-mode / sandbox / internal results are NEVER real outcomes — label them
test-mode and keep them separate from real ones.

## Real actions only — never simulate, never fake progress

Progress is measured ONLY by real, external, verifiable OUTCOMES — an email
actually accepted by the SMTP server, a real URL returning 200, a payment that
settled, a meeting on the calendar. Writing a file, drafting, planning, or
"preparing" an action is NOT progress and NEVER counts as having done it. A team
that produces plans and files but takes no real action has failed, however busy
it looks.

- **NEVER simulate, mock, dry-run, or "log it as if done."** If the task is a
  real action, perform the real action or do not claim it. A results file full
  of `"ok": true` that no real call produced is a lie to your team. This rule
  OVERRIDES any role instruction that says "if you lack a tool, simulate it" —
  there is no such fallback; simulation is never an acceptable substitute.
- **Blocked from the real action?** (no tool, no credential, no permission, or
  the real step itself fails) → STOP and escalate to the human ONCE (ask_human /
  request_human_takeover), then switch tasks. Do NOT fall back to file/plan
  busywork to look productive.
- **Prove capability FIRST.** If a deliverable's final step needs an unproven
  ability (an SMTP send, a logged-in account, a payment method), prove it with
  ONE cheap REAL test before building toward it. If it's missing → escalate now
  and switch tasks. Preparing 10 files to send 0 emails is this swarm's most
  expensive recorded failure — never build toward an action you cannot execute.
- **Real actions need real inputs.** Never fire a real action on guessed or
  unverified data (e.g. a guessed email address, a placeholder id). Verify the
  input first, or escalate — a real send to a fabricated address is still a
  failure, not an action.

## You are fully autonomous

Root shell, filesystem, terminal, browser, deploys, databases — anything an
operator could do, you do yourself, now, without asking. Humans exist ONLY for:
credentials / 2FA / CAPTCHA (request_human_takeover), authorizing real spend,
or sign-off on an irreversible step (ask_human). Never ask
a human to run commands, read files, or verify URLs for you. Act only on real
queued messages and the brief — never on an imagined directive.

**Human inbox messages are read on a phone — keep them SHORT.** A human asks
get ≤4 lines: the decision you need and the options, nothing else. No preamble,
no recap of what you did, no file dumps, no pasted logs. Lead with the ask. Light
markdown (a **bold** phrase, a short bullet list) is rendered, so use it to make
the one question scannable — but brevity beats formatting. If you feel the urge
to explain context, put it in a log_decision instead, not the human's pocket.

Missing or weak tool? Build your own: diagnose the error, web_search how others
solve it, write a script under tools/ with a README line, test it from the
terminal, use it, commit it. Check tools/README.md first — a teammate may have
already built it.

## Messaging — `kind` decides everything (platform-enforced)

| kind | wakes them? | use for |
|------|-------------|---------|
| TASK / QUESTION | YES — they owe you a RESULT (you get a task_id) | delegating work / a question you need answered |
| RESULT | wakes only the delegator and closes the ledger item | finished work — always pass reply_to=<task_id> |
| STATUS / FYI | NO — shown quietly at the start of their next turn | progress notes, heads-ups |

- Delegate by role-fit to ONE peer. Report results UP to your delegator — never
  bounce work sideways.
- Put the FULL spec in the FIRST task (format, count, destination, constraints).
  Follow-up re-dos regenerate the whole artifact — if a deliverable is good
  enough, use it.
- ONE canonical file per deliverable: search_files first, then extend or
  overwrite it. Never spawn variants (x_v2, x_fixed, x_final).
- No concrete ask → STATUS/FYI, or send nothing. NEVER a TASK/QUESTION to
  acknowledge, confirm, or re-state something unchanged.
- You may message ONLY: {allowed_peers_list}.

## Where facts live — SHARED vs PRIVATE

| store | scope | for |
|-------|-------|-----|
| workspace.md (edit with write_file) | SHARED | the brief / goals / conventions — the single source of truth |
| log_decision | SHARED | ONE line others must know; auto-injected to everyone next turn; no trivia |
| log_action | SHARED | call BEFORE any external side-effect (send / publish / deploy / pay) with a stable idempotency_key — duplicate=true means a peer already did it: use their outcome, do NOT repeat it |
| memory tool | PRIVATE | your own scratch; peers NEVER see it |

If a teammate needs a fact, it goes in a SHARED store — writing it to private
memory is the same as telling no one.

## Shared workspace & production safety

- One shared project dir for the whole team: {project_dir}. One git branch —
  commit small coherent units; use `git worktree add` to isolate big changes;
  message a peer before touching a file they are working in.
- **Lease before mutating production or driving the shared browser**: write
  /tmp/prod_lock (or /tmp/browser_lock) containing `agent={agent_name} ts=<now>`,
  wait 2s, re-read it. Not your name → the lease is taken: back off and retry
  later. Delete the lock when done. Can't acquire within 10s → log the conflict
  and do other work first.

## Terminal

Starts in {project_dir}. The working directory does NOT persist between calls —
chain `cd sub && …` or use absolute paths. Long-running process →
background=true, then health-check it with a follow-up command.

## Browser — climb this ladder, don't repeat a failing rung

1. Refs first: browser_snapshot → browser_click / browser_type on @eN.
2. Rich-text editors (Medium/Notion-style) and keyboard shortcuts: click/focus
   the area, then browser_keys — REAL keystrokes into the focused element
   (browser_type's clear+fill silently fails in these editors). browser_keys
   is multi-line: ONE call types a whole document (\n = real Enter), and
   from_file="docs/post.md" types a file's entire contents — NEVER type an
   article one paragraph per call.
3. Already know the next 2-10 actions (click → type → press, fill a small
   form)? ONE browser_steps call runs the whole sequence and stops at the
   first failure — never spend N separate calls on a flow you can script.
4. Hover-only menus → browser_hover. Reordering/sliders → browser_drag. File
   pickers → browser_upload (never click through the OS dialog). Slow
   saves/loads → browser_wait, not blind re-clicks.
5. Visible on screen but NO usable ref (canvas, unlabeled icons, custom
   widgets)? browser_locate("describe it visually") → browser_click_xy(x,y).
6. SEE before you claim: after any step that should visibly change the page,
   browser_screenshot (saved to your workspace — cite the path in your RESULT)
   or browser_vision to confirm it actually happened.
The same action failing twice will not pass on try 3 — move DOWN the ladder.
Every action result includes page_alerts — the page's own visible error text.
Read it before re-trying anything; never hunt for an error with console scans.

## Credentials & logins

- list_credentials → what the team has; get_credential(site) → username +
  secret + its PURPOSE. Use a credential ONLY for its stated purpose — an SMTP
  app password is not a website login, an API key is not an email password.
- No stored credential for the account you need? STOP guessing. One
  request_human_takeover ("Log in to <site>") hands the human this exact
  browser; the session persists for every agent afterwards. Same for any
  CAPTCHA or verification code — never retry through one.
- Never type a secret into a site it wasn't stored for, and never paste
  secrets into messages, files, or logs.
- NEVER inline a secret value into code or a file you write — always
  `os.environ.get(...)` / read it from .env at runtime. Inlined secrets are
  auto-redacted from your own history, so the file you wrote will look
  "corrupted"/truncated to you afterwards even though it is fine on disk —
  do NOT loop rewriting it. (Disk state is ground truth; your echo is
  redacted.)

## Sub-agents for grindy work

delegate_task spawns a fresh sub-agent that does a job and returns ONLY a
summary — its tool calls never enter your context, which makes a long
mechanical job ~10x cheaper there than in your own turn.
- USE IT when a job is 5+ mechanical tool steps: scraping a list of pages,
  filling repetitive forms, bulk file edits, a multi-page research sweep.
- The sub-agent knows NOTHING about the team or mission — put everything it
  needs (URLs, file paths, exact success criteria, what to return) in the
  goal/context, pass narrow toolsets (e.g. ["browser","file"]), and set
  max_iterations.
- NEVER use it to talk to teammates (send_peer_message is the only channel) or
  for judgment calls that need team state — those are yours.

## Economy

- Need 2+ files? ONE read_files(paths=[...]) call — never a chain of single
  read_file calls. Every extra tool round trip re-sends your ENTIRE context
  to the model.
- Never re-read a file you read earlier this turn unless you changed it.
- 5+ known mechanical steps → ONE browser_steps / browser_keys(from_file) /
  delegate_task call, never N separate round trips.
- web_search to find information; the browser only to open a URL you already
  have.
- Don't re-read a file you just wrote unless the next step needs its contents.

## Self-management

- get_self_config → your model, tools, context usage, spend.
  request_config_change → propose a settings change (a human approves it).
- schedule_wakeup / cancel_wakeup → your own recurring wake-ups. Each
  instruction must be self-contained; check your existing schedules in LIVE
  CONTEXT before creating one.
"""


# ---------------------------------------------------------------------------
# Task-injection prompts — wrap a message/transcript when it is enqueued as a
# task. These are user-message-side (not part of the system prompt).
# ---------------------------------------------------------------------------
AUTONOMOUS_HEARTBEAT_PROMPT = (
    "[IDLE HEARTBEAT — automated check-in; you run 24/7 and no human is watching]\n"
    "Current Time: {time}\n"
    "Your queue is empty. Re-read the PROJECT BRIEF, OUTSTANDING WORK and RECENTLY "
    "COMPLETED in LIVE TEAM CONTEXT, then do exactly ONE of these and end your turn:\n"
    "• A concrete, role-fitting next step toward the brief exists → take it (or "
    "delegate it to the right peer) NOW, ending in the tool call that does it. If "
    "you lead others, prefer unblocking/feeding a report who is idle.\n"
    "• Every real next step is blocked on a human (see WAITING ON HUMAN) → do NOT "
    "invent substitute busywork and do NOT re-ask; end your turn with no tool call. "
    "You are re-woken the moment the human answers.\n"
    "• The mission is genuinely complete, or nothing remains that fits your role → "
    "log_decision that conclusion ONCE (skip if already logged), then end. Idle is "
    "correct: while you stay idle these check-ins automatically become less "
    "frequent, and any real message resets them.\n"
)

# Injected (at most ONCE per occurrence — capped by AgentDaemon._text_only_nudged)
# when a turn ended with a text-only summary and no committal tool call: the agent
# "ended in the chat" instead of acting. Convert that wasted turn into a real action.
TEXT_ONLY_TURN_NUDGE = (
    "[LOST OUTPUT — your last turn ended in free text with no committing tool call]\n"
    "That closing text reached no teammate and changed nothing (only tool calls "
    "deliver). A status summary is not a turn. Do exactly ONE of these now, then stop:\n"
    "• If there is a next action — delegate it with `send_peer_message` (TASK), or "
    "publish/deploy/send it with the tool that does it.\n"
    "• If you only established a fact others need — record it with ONE `log_decision`.\n"
    "• If there is genuinely nothing left to do — end with a SINGLE `log_decision` "
    "noting the conclusion, or no output at all.\n"
    "Do NOT reply with another summary or 'If you want, I can…'. Act or record — once."
)

# Idle check-in while a TIME-BOXED DIRECTIVE is active. A normal heartbeat
# lets the agent conclude "all tasks closed → idle is correct"; a directive
# explicitly is not satisfied by an empty queue. (Observed: a founder given a
# 4-hour push closed it out in 25 minutes and idled the rest.) Re-presents
# the directive verbatim with remaining time until it expires; heartbeat
# backoff is suspended while one is active.
DIRECTIVE_HEARTBEAT_PROMPT = (
    "[ACTIVE DIRECTIVE — {remaining} remaining; check-in, queue empty]\n"
    "Current Time: {time}\n"
    "You are mid-way through a time-boxed directive from {from_agent}. An empty "
    "queue does NOT complete it — closed sub-tasks only mean the last increment "
    "landed. The directive, verbatim:\n"
    "---\n{directive}\n---\n"
    "Review what has shipped so far (RECENTLY COMPLETED, decisions), then start "
    "the next highest-leverage increment toward it NOW — do it or delegate it, "
    "ending in the tool call that does it. Only two exceptions: every avenue is "
    "blocked on a human (end with no tool call), or you can argue the directive "
    "is genuinely exhausted — then log_decision that argument ONCE and idle."
)

# Injected ONCE per delegated task when the worker's turn on a peer/human
# TASK/QUESTION ended without ANY send_peer_message — the silent-stall mode:
# the worker ran out of iterations (or just stopped) and its progress lives
# only in its own conversation, invisible to the delegator. (Observed: a
# worker stalled twice this way for 95 minutes until a supervisor sweep
# happened to chase it.) One forced status RESULT converts the silent stall
# into a cheap, honest handoff.
MISSING_RESULT_NUDGE = (
    "[TURN ENDED WITHOUT A RESULT — task {task_id} from {from_agent}]\n"
    "Your last turn on this task ended (iteration budget or stop) without "
    "sending ANY message back. The delegator cannot see your conversation — "
    "only send_peer_message delivers. Send ONE concise status RESULT now:\n"
    "  send_peer_message(to_agent=\"{from_agent}\", kind=\"RESULT\", "
    "reply_to=\"{task_id}\", message=<DONE: …  REMAINING: …  NEXT: …>)\n"
    "If the work is incomplete, say exactly what remains — a partial status is "
    "infinitely better than silence. Do nothing else this turn."
)

# Injected ONCE (cooldown-limited) when the daemon's repetition guard sees the
# same tool call with the same arguments recur across several SEPARATE turns —
# the single-agent self-loop that the pair/team detectors cannot see. Names the
# exact repeated call so the agent can't mistake which behavior to change.
SELF_LOOP_NUDGE = (
    "[SYSTEM — REPETITION GUARD] Across your recent turns you repeated the same "
    "tool call with the same arguments:\n"
    "  {signature}\n"
    "Repeating it again will not produce a new result. Do exactly ONE of these "
    "now, then stop:\n"
    "• Try a genuinely DIFFERENT approach to the same goal (different tool, "
    "different arguments, or build a small script under tools/).\n"
    "• If this was a check that keeps failing or never changes: log_decision the "
    "state as BLOCKED/UNVERIFIED with the exact error, then switch to other work "
    "(escalate once with ask_human only if a human is truly required).\n"
    "• If this was re-verification of something already settled: accept the "
    "recorded result and move on.\n"
    "Do not reply to or acknowledge this notice."
)

# Injected as a task when a per-agent cron wake-up fires (see AgentDaemon._maybe_fire_crons
# and the schedule_wakeup tool). Unlike the heartbeat, a cron carries a SPECIFIC
# instruction the agent (or operator) attached when scheduling it.
CRON_WAKEUP_PROMPT = (
    "[SCHEDULED WAKE-UP — cron '{schedule}' fired at {time}; scheduled "
    "{scheduled_ago} ago; you run 24/7 and nobody may be watching]\n"
    "This is an automated wake-up you or your operator scheduled. The instruction "
    "below states the GOAL — it is not a frozen script. It was written "
    "{scheduled_ago} ago and the world has moved since: re-derive the right steps "
    "from CURRENT state (files, data, recent messages), and if it references a "
    "runbook/doc, follow that file's latest version. It never overrides fresher "
    "directives from a human or your supervisor. If it has become obsolete or "
    "superseded, log_decision why and fix the schedule itself (schedule_wakeup / "
    "cancel_wakeup) instead of executing it on autopilot.\n"
    "Carry out the instruction, then end your turn (do not loop):\n\n"
    "{instruction}"
)

# The periodic sweep delivered to a supervisor: EVERYTHING its linked agents
# did in the last window (live — a mid-turn agent shows its partial turn so
# far), one section per agent with a CODE-COMPUTED progress signal, plus a
# team ledger (states, open delegations with ages, human blocks). Silent
# agents get an explicit "(no activity)" section — silence with owed work is
# a finding, not a gap in the data.
SUPERVISOR_SWEEP_PROMPT = (
    "[SUPERVISOR SWEEP — automated, every ~{window_minutes} min; you watch "
    "{peer_count} agent(s)]\n"
    "Below is everything each of your agents did since your last sweep. An "
    "agent marked BUSY may be MID-TURN — its section shows the partial turn so "
    "far. Each section starts with a computed signal; trust it over your "
    "impression. Compare against your PREVIOUS sweep (earlier in this "
    "conversation): what actually moved since then?\n\n"
    "=== LEDGER ===\n{ledger}\n\n{sections}\n"
    "=== HOW TO RESPOND ===\n"
    "Judge each agent: shipped / busy-on-mission / stalled-or-looping / silent "
    "with owed work / fine-idle. Then act, sparingly:\n"
    "- Healthy (shipping, or idle with nothing owed): do NOT message it.\n"
    "- Stalled, looping, or off-brief: ONE short, directive "
    "send_peer_message naming the single concrete next action to take this "
    "turn (a publish, deploy, send, fix, measured result), or the exact "
    "blocker to clear. One message per problem agent, maximum.\n"
    "- SILENT but owing an open delegation (see LEDGER ages): wake the OWNER "
    "with one message naming that delegation — do not bother the delegator.\n"
    "- Two agents waiting on each other / re-trading an unchanged status: do "
    "not message INTO the loop. If you steered the same problem last sweep "
    "and it recurs, call `pause_agent` on the looping agent(s), log_decision "
    "why, and ask_human once if an operator is needed.\n"
    "- Blocked on a human (LEDGER shows the ask): do not nag the agent; "
    "ask_human once if the question has sat unanswered across sweeps.\n"
    "- NEVER acknowledge, 'noted', or status-echo — to anyone, ever.\n"
    "Close EVERY sweep with exactly ONE log_decision line summarizing your "
    "verdict (e.g. 'sweep: 4 ok, nudged X re: stuck deploy') — even when all "
    "is well — then end your turn."
)

# Default identity for an agent created as a supervisor (used when the operator
# doesn't supply their own role_soul).
SUPERVISOR_DEFAULT_SOUL = (
    "You are a SUPERVISOR agent. You do NOT do project work yourself. On a fixed "
    "interval, a SWEEP lands in your queue automatically (you do not fetch it): "
    "everything every agent you watch did in the window — including partial "
    "mid-turn activity — one section per agent with a CODE-COMPUTED progress "
    "signal, plus a ledger of agent states, open delegations with ages, and "
    "human blocks. Agents with NO activity are listed too: silence while owing "
    "an open delegation is a finding, not a gap.\n"
    "Your one job is to keep them PROGRESSING — shipping concrete actions that move "
    "the mission — not merely busy. Each sweep tells you how many real external "
    "actions each agent took versus how many turns were near-duplicate "
    "re-confirmations; compare with your previous sweep to see what actually moved.\n"
    "WHAT IS DRIFT: looping or repeating a tool call; re-confirming/acknowledging an "
    "unchanged status; burning turns with ZERO concrete action; re-verifying work "
    "already done; busywork that does not advance the team's brief; silently blocked; "
    "two teammates duplicating; or risky/destructive actions. A polite agent calmly "
    "re-acknowledging the same status every turn IS drift — the most common and most "
    "expensive kind. Treat it as a problem, not as healthy.\n"
    "WHEN YOU SEE DRIFT: steer with ONE short, specific send_peer_message naming the "
    "single concrete next action the agent must take this turn. Be directive.\n"
    "WHEN STEERING ISN'T ENOUGH: if you already steered and the same loop recurs, or "
    "two peers are stuck pinging each other (re-asking for the same proof, trading an "
    "unchanged status), do NOT send another message into it — call `pause_agent` on "
    "the looping agent(s) to physically break the loop, log_decision why, and "
    "ask_human once if an operator is needed. A degenerate loop is paused, not "
    "chatted at.\n"
    "NEVER ACK BACK: do not reply to a reviewee with 'noted' / 'acknowledged' / a "
    "status echo — that adds to the loop you exist to break. Your only two valid "
    "outputs are (a) a corrective steer to an agent that needs one, and (b) the "
    "single terse log_decision verdict that closes every sweep (even an all-well "
    "one). A peer message is ONLY for steering.\n"
    "Be sparing and high-signal — intervene rarely but decisively, and never join "
    "the chatter you are meant to police."
)


# ---------------------------------------------------------------------------
# Team-context builders
# ---------------------------------------------------------------------------
def _build_peer_roster(cfg: Dict[str, Any], team_id: str, current_agent: str) -> str:
    """Compact org context for ONE agent: only its directly-linked peers (with
    one-line roles) plus a one-line team-size summary.

    Replaces the old full-team org diagram + member list, which dumped every
    agent's adjacency into EVERY agent's prompt — O(N²) tokens that scaled to
    ~4k/turn on a 43-agent team. An agent can only message its linked peers, so
    the peers' roles are all the org context it actually needs to route work; the
    full roster lives in the changelog / live-context panel for awareness."""
    agents = cfg.get("agents", {})
    team_agents = {n: a for n, a in agents.items() if a.get("team_id") == team_id}
    me = team_agents.get(current_agent, {})
    peers = me.get("allowed_peers", []) or []

    def _role_of(name: str) -> str:
        a = agents.get(name, {})
        return a.get("role_soul", f"You are the {a.get('name', name)}.").split("\n")[0][:80]

    lines = [f"Team '{team_id}': {len(team_agents)} agents total (tree topology; "
             f"messaging reaches ONLY your linked peers below)."]
    if peers:
        lines.append("Your directly-linked peers (the only agents you can message):")
        for p in peers:
            lines.append(f"  - {p}: {_role_of(p)}")
    else:
        lines.append("You currently have no linked peers.")
    return "\n".join(lines)


def compose_soul_identity(agent_cfg: Dict[str, Any]) -> str:
    """Build the SOUL.md identity block for an agent.

    This becomes the agent's PRIMARY identity, written to {HERMES_HOME}/SOUL.md
    so Hermes loads it as the lead block of the (cached) system prompt — instead
    of the generic auto-seeded "You are Hermes Agent…" template. The richer
    operational framing (swarm rules, org chart, peers) stays in the ephemeral
    prompt via compose_agent_soul(..., include_role=False); the role lives here
    so it is the first thing the model reads and is cache-stable across turns.
    """
    agent_name = agent_cfg.get("name", "Agent")
    agent_id = agent_cfg.get("agent_id", "unknown")
    team_id = agent_cfg.get("team_id", "default")
    role = agent_cfg.get("role_soul", f"You are the {agent_name}.")
    return (
        f"{role}\n\n"
        f'You are "{agent_id}" ({agent_name}), one agent on the "{team_id}" team in an '
        f"autonomous multi-agent swarm. Operate strictly in the role described above."
    )


def _read_workspace_brief(team_id: str, max_chars: int = 8000) -> str:
    """Return the team's workspace.md text for inlining into the prompt.

    Truncated defensively so an oversized brief can't blow up every turn's
    token budget. Returns a friendly placeholder if the file is absent."""
    try:
        p = _get_team_workspace_path(team_id) / "workspace.md"
        if not p.exists():
            return "(no workspace.md yet — the team brief has not been written.)"
        text = p.read_text(encoding="utf-8").strip()
        if len(text) > max_chars:
            text = text[:max_chars] + "\n…(brief truncated — read workspace.md on disk for the rest.)"
        return text or "(workspace.md is empty.)"
    except Exception as e:
        return f"(could not read workspace.md: {e})"


def _build_workspace_tree(team_id: str, max_entries: int = 160) -> str:
    """A compact directory listing of the team's SHARED project dir, so each
    agent can SEE what exists in the one shared repo without guessing paths.
    Skips noise (.git, .hermes, caches, browser profile) and caps total lines so
    a large repo can't dominate the prompt."""
    import os
    root = _get_project_dir(team_id)
    if not root.exists():
        return "(shared project not created yet.)"
    SKIP = {".git", ".hermes", "__pycache__", "node_modules", ".browser-profile",
            "context", ".DS_Store", "dist", "build", ".venv"}
    lines: List[str] = []
    truncated = False
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP)
        rel = os.path.relpath(dirpath, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth > 4:
            dirnames[:] = []
            continue
        indent = "  " * depth
        label = "." if rel == "." else os.path.basename(dirpath)
        lines.append(f"{indent}{label}/")
        for fn in sorted(filenames):
            if fn.endswith((".pyc", ".db", ".db-shm", ".db-wal")) or fn == ".DS_Store":
                continue
            lines.append(f"{indent}  {fn}")
            if len(lines) >= max_entries:
                truncated = True
                break
        if truncated:
            break
    if truncated:
        lines.append("…(tree truncated)")
    return "\n".join(lines)


def _recent_peer_messages(team_id: str, full_config: Optional[Dict[str, Any]], limit: int = 10) -> str:
    """Render the last N send_peer_message events across the team, oldest→newest,
    so every agent has shared awareness of recent team chatter."""
    try:
        from swarm_server.monitoring import monitor_db
        # message_sent events aren't team-tagged, so scope by the team's agent set.
        team_agents = set()
        if full_config:
            team_agents = {
                aid for aid, a in (full_config.get("agents") or {}).items()
                if a.get("team_id") == team_id
            }
        events = monitor_db.get_events(limit=200)  # newest first
        msgs = []
        for e in events:
            if e.get("event_type") != "message_sent":
                continue
            frm, to = e.get("from_agent"), e.get("to_agent")
            if team_agents and frm not in team_agents and to not in team_agents:
                continue
            preview = kind = ""
            try:
                d = json.loads(e.get("data") or "{}") or {}
                preview = d.get("message_preview", "")
                kind = d.get("kind", "")
            except Exception:
                pass
            msgs.append((e.get("timestamp", 0), frm, to, preview, kind))
            if len(msgs) >= limit:
                break
        if not msgs:
            return "(no peer messages yet.)"
        msgs.reverse()  # oldest → newest reads naturally
        out = []
        for ts, frm, to, preview, kind in msgs:
            stamp = ""
            try:
                from datetime import datetime
                stamp = datetime.fromtimestamp(ts).strftime("%H:%M")
            except Exception:
                pass
            tag = f"{kind} " if kind else ""
            out.append(f"  [{stamp}] {frm} → {to} ({tag.strip() or 'MSG'}): {preview}")
        return "\n".join(out)
    except Exception as e:
        return f"(could not load peer messages: {e})"


def _age_str(ts: float) -> str:
    """Compact human age for ledger lines: '4m', '2.3h', '1.5d'."""
    try:
        import time as _t
        secs = max(0.0, _t.time() - float(ts or 0))
    except Exception:
        return "?"
    if secs < 90:
        return f"{int(secs)}s"
    if secs < 5400:
        return f"{int(secs // 60)}m"
    if secs < 129600:
        return f"{secs / 3600:.1f}h"
    return f"{secs / 86400:.1f}d"


def _open_delegations_block(team_id: str, agent_id: str) -> str:
    """What this agent owes and is owed, from the delegation ledger — so an agent
    (especially a coordinator) sees outstanding work without polling peers.
    Each item carries its age; items older than 2h are flagged so a stuck
    delegation is visibly stuck instead of silently rotting in the list."""
    try:
        from swarm_server.monitoring import monitor_db
        owe = monitor_db.get_open_delegations(to_agent=agent_id, team_id=team_id, limit=20)
        awaiting = monitor_db.get_open_delegations(from_agent=agent_id, team_id=team_id, limit=20)
        if not owe and not awaiting:
            return "(nothing outstanding.)"

        def _line(d, who):
            age = _age_str(d.get("timestamp", 0))
            overdue = ""
            try:
                import time as _t
                if _t.time() - float(d.get("timestamp", 0)) > 7200:
                    overdue = " ⚠ overdue — follow up ONCE or reassign; do not just wait"
            except Exception:
                pass
            return f"    - id={d['msg_id']} {who} ({age} ago): {d.get('summary','')}{overdue}"

        lines = []
        if owe:
            lines.append("  YOU OWE A RESULT (open TASK/QUESTION sent to you):")
            for d in owe:
                lines.append(_line(d, f"from {d['from_agent']}"))
        if awaiting:
            lines.append("  AWAITING A RESULT (you delegated, not yet answered):")
            for d in awaiting:
                lines.append(_line(d, f"to {d['to_agent']}"))
        lines.append(
            "  An entry you have VERIFIED is already delivered/obsolete via another "
            "chain: close it ONCE with close_ledger_entry(msg_id, reason) — never "
            "re-analyze the same stale entry on a later check-in.")
        return "\n".join(lines)
    except Exception as e:
        return f"(could not load delegations: {e})"


def _recently_completed_block(team_id: str, limit: int = 8) -> str:
    """The last N CLOSED delegations (RESULT received), newest first.

    This is the anti-re-ask ledger: before delegating, an agent can see that the
    same work was already delivered and by whom. (Observed failure without it: a
    coordinator re-delegated an already-delivered prospect list three times.)"""
    try:
        from swarm_server.monitoring import monitor_db
        rows = monitor_db.get_recent_closed_delegations(team_id=team_id, limit=limit)
        if not rows:
            return "(none yet.)"
        lines = []
        for d in rows:
            closed = _age_str(d.get("answered_at") or d.get("timestamp", 0))
            lines.append(
                f"  - id={d['msg_id']} {d['from_agent']}→{d['to_agent']} "
                f"(closed {closed} ago): {d.get('summary','')}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"(could not load completed delegations: {e})"


def _waiting_on_human_block(team_id: str, full_config: Optional[Dict[str, Any]]) -> str:
    """Team-wide list of UNANSWERED ask_human / takeover requests.

    Before this block, only the asking agent knew it was blocked on a human —
    its delegators kept assigning work whose final step was behind that same
    locked door (observed: a full run spent preparing emails while the SMTP
    credential ask sat unanswered). Surfacing the pending asks to the whole
    team lets everyone route around the blocked path instead of into it."""
    try:
        from swarm_server.tools import get_pending_questions
        team_agents = None
        if full_config:
            team_agents = {
                aid for aid, a in (full_config.get("agents") or {}).items()
                if a.get("team_id") == team_id
            }
        pending = [
            q for q in get_pending_questions()
            if q.get("status") == "pending"
            and (team_agents is None or q.get("agent_name") in team_agents)
        ]
        if not pending:
            return "(nothing — no unanswered human requests.)"
        pending.sort(key=lambda q: q.get("timestamp", 0))
        lines = []
        for q in pending[:6]:
            ask = " ".join((q.get("question") or "").split())[:110]
            lines.append(
                f"  - {q.get('agent_name')} asked {_age_str(q.get('timestamp', 0))} ago: {ask}"
            )
        if len(pending) > 6:
            lines.append(f"  …and {len(pending) - 6} more.")
        lines.append(
            "  → Work whose FINAL step depends on one of these is BLOCKED: do not "
            "start/delegate it, and do not re-ask the human."
        )
        return "\n".join(lines)
    except Exception as e:
        return f"(could not load pending human requests: {e})"


def _build_cron_summary(full_config: Optional[Dict[str, Any]], agent_id: str) -> str:
    """One line per scheduled wake-up this agent currently has, for the prompt."""
    if not full_config:
        return "(unavailable)"
    crons = (full_config.get("agents", {}).get(agent_id, {}) or {}).get("crons") or []
    if not crons:
        return "(none — you have no scheduled wake-ups)"
    try:
        from swarm_server.cron import cron_describe
    except Exception:
        cron_describe = lambda s: s  # noqa: E731
    lines = []
    for c in crons:
        state = "enabled" if c.get("enabled", True) else "disabled"
        instr = (c.get("instruction") or "").replace("\n", " ")
        if len(instr) > 100:
            instr = instr[:100] + "…"
        lines.append(
            f"- [{state}] {c.get('schedule')} ({cron_describe(c.get('schedule', ''))})"
            f" → {instr}  (id={c.get('id')})"
        )
    return "\n".join(lines)


def _recent_decisions(team_id: str, limit: int = 20) -> str:
    """Render the team's last N decisions (oldest→newest) for prompt injection.

    This is the durable team memory that replaced agent_log.md: agents append
    one-liners via the log_decision tool and read them back here every turn, so
    the team stays coordinated without anyone opening a file."""
    try:
        from swarm_server.monitoring import monitor_db
        rows = monitor_db.get_recent_decisions(team_id, limit=limit)  # newest first
        out = []
        # Older decisions are rolled up into a milestone so long-term memory isn't
        # lost when entries scroll past the live window — show it first.
        try:
            ms = monitor_db.get_latest_milestone(team_id)
            if ms and ms.get("summary"):
                out.append(f"  [EARLIER — rolled-up milestone] {ms['summary']}")
        except Exception:
            pass
        if not rows and not out:
            return "(no decisions logged yet.)"
        for r in reversed(rows):  # oldest → newest reads naturally
            stamp = ""
            try:
                from datetime import datetime
                stamp = datetime.fromtimestamp(r.get("timestamp", 0)).strftime("%m-%d %H:%M")
            except Exception:
                pass
            out.append(f"  [{stamp}] {r.get('agent_name')}: {r.get('decision')}")
        return "\n".join(out)
    except Exception as e:
        return f"(could not load decisions: {e})"


# Sentinels around every injected live-context block. They make the block
# machine-findable so the daemon can strip EXPIRED copies out of replayed
# history: without stripping, a 30-turn session carries 30 slightly-different
# "current" snapshots of the brief/ledger/decision log, and the model has no
# reliable way to tell which one is true — observed as agents acting on stale
# ledger state (re-delegating closed work) and as the single largest token
# line item (multi-KB block re-stored in every user turn forever).
LIVE_CONTEXT_BEGIN = "<<<LIVE-CONTEXT>>>"
LIVE_CONTEXT_END = "<<<END-LIVE-CONTEXT>>>"

# What an expired snapshot is replaced with when history is replayed.
STALE_CONTEXT_NOTE = (
    "[expired team-context snapshot removed — the CURRENT one is in the latest "
    "message]"
)

# Legacy (pre-sentinel) live-context headers, for stripping old stored sessions.
_LEGACY_CTX_HEADERS = (
    "--- LIVE TEAM CONTEXT (auto-refreshed each turn) ---",
    "--- LIVE CONTEXT (auto-refreshed each turn) ---",
)


def strip_stale_live_context(text: str) -> str:
    """Remove an embedded live-context snapshot from ONE historical message.

    Sentinel-delimited blocks (current format) are replaced with
    STALE_CONTEXT_NOTE. Legacy blocks (no sentinels) are stripped only when the
    known batch header ("You have N new message(s)") follows, so a real task
    body can never be eaten by accident. Idempotent; returns text unchanged on
    any doubt."""
    import re as _re
    if not text:
        return text
    out = text
    if LIVE_CONTEXT_BEGIN in out:
        if LIVE_CONTEXT_END in out:
            out = _re.sub(
                _re.escape(LIVE_CONTEXT_BEGIN) + r".*?" + _re.escape(LIVE_CONTEXT_END),
                STALE_CONTEXT_NOTE, out, flags=_re.DOTALL,
            )
        else:
            # Truncated block (e.g. by a char cap): fall back to the batch anchor.
            m = _re.search(r"You have \d+ new message", out)
            if m:
                start = out.index(LIVE_CONTEXT_BEGIN)
                if start < m.start():
                    out = out[:start] + STALE_CONTEXT_NOTE + "\n\n" + out[m.start():]
    else:
        for hdr in _LEGACY_CTX_HEADERS:
            if hdr in out:
                m = _re.search(r"You have \d+ new message", out)
                start = out.index(hdr)
                if m and start < m.start():
                    out = out[:start] + STALE_CONTEXT_NOTE + "\n\n" + out[m.start():]
                break
    return out


# Marker prefix for an aged-out tool result. Doubles as the idempotency guard:
# a stub starts with this and is far shorter than any min_chars threshold, so
# it can never be re-stubbed.
TOOL_RESULT_ELIDED_PREFIX = "[stale tool result elided"


def age_stale_tool_results(
    messages: List[Dict[str, Any]],
    keep_recent: int,
    min_chars: int,
    quantum: int,
) -> List[Dict[str, Any]]:
    """Replace OLD, LARGE tool results in replayed history with one-line stubs.

    Tool results are 50-60% of a working agent's history (measured), and every
    LLM iteration re-bills the whole history — yet a 3-day-old browser snapshot
    or file dump is almost never read again, and if it IS needed the agent can
    just re-run the tool. Unlike compaction this deletes nothing the agent SAID
    or DECIDED; assistant/user turns stay verbatim.

    Rules:
    - The most recent `keep_recent` messages are never touched (the working set).
    - The cutoff is quantized: it only advances every `quantum` messages, so the
      request prefix stays byte-stable between steps (provider prompt caches
      survive ~quantum turns instead of busting every turn).
    - Only role=="tool" messages with a str content longer than `min_chars` are
      stubbed; the stub names the tool and original size so the model knows what
      was there and how to get it back. tool_call_id/name fields are preserved
      so pairing with the assistant's tool_calls stays intact.
    - Idempotent; returns the original list object if nothing changed.
    """
    n = len(messages)
    if n <= keep_recent or keep_recent < 0 or quantum <= 0:
        return messages
    cutoff = ((n - keep_recent) // quantum) * quantum
    if cutoff <= 0:
        return messages
    out: List[Dict[str, Any]] = []
    changed = False
    for i, m in enumerate(messages):
        if (
            i < cutoff
            and m.get("role") == "tool"
            and isinstance(m.get("content"), str)
            and len(m["content"]) > min_chars
            and not m["content"].startswith(TOOL_RESULT_ELIDED_PREFIX)
        ):
            name = m.get("name") or "tool"
            stub = (
                f"{TOOL_RESULT_ELIDED_PREFIX} — {name}, was "
                f"{len(m['content'])} chars; re-run the tool if you need it]"
            )
            m = {**m, "content": stub}
            changed = True
        out.append(m)
    return out if changed else messages


SWEEP_PAYLOAD_MARKER = "[SUPERVISOR SWEEP"
SWEEP_PAYLOAD_ELIDED_STUB = (
    "[old sweep payload elided — that window's full team feed is stale; your "
    "verdict for it follows below. Only the LATEST sweep payload is actionable.]"
)


def age_stale_sweep_payloads(
    messages: List[Dict[str, Any]],
    keep_recent: int,
    min_chars: int,
) -> List[Dict[str, Any]]:
    """Replace OLD supervisor-sweep payloads in replayed history with a stub.

    A supervisor's history is dominated by its own past sweep inputs — measured
    96% of one overseer's session chars were old `[SUPERVISOR SWEEP …]` team
    feeds (the *user* side, which tool-result aging never touches) — and every
    sweep iteration re-bills all of them. An old feed is pure history: the
    verdict the supervisor wrote in response (kept verbatim, right after it)
    is the durable record.

    Only the last `keep_recent` payload-bearing user messages stay intact.
    Older ones are truncated FROM THE MARKER ONWARD (any live-context prefix is
    handled separately by strip_stale_live_context), provided the elided tail
    exceeds `min_chars`. The stub does not contain the marker, so the pass is
    idempotent and stubbed messages stop counting toward `keep_recent`.
    Returns the original list object if nothing changed."""
    if keep_recent < 0 or not messages:
        return messages
    payload_idx = [
        i for i, m in enumerate(messages)
        if m.get("role") == "user"
        and isinstance(m.get("content"), str)
        and SWEEP_PAYLOAD_MARKER in m["content"]
    ]
    stale = payload_idx[:-keep_recent] if keep_recent else payload_idx
    if not stale:
        return messages
    out = list(messages)
    changed = False
    for i in stale:
        content = out[i]["content"]
        pos = content.index(SWEEP_PAYLOAD_MARKER)
        if len(content) - pos <= min_chars:
            continue
        out[i] = {**out[i], "content": content[:pos] + SWEEP_PAYLOAD_ELIDED_STUB}
        changed = True
    return out if changed else messages


def compose_live_context(
    team_id: str,
    agent_id: str,
    full_config: Optional[Dict[str, Any]] = None,
) -> str:
    """Dynamic, per-turn context prepended to the final user message: brief,
    shared tree, decision log, open + recently-closed delegations, pending
    human asks, recent chatter, crons. Rebuilt each turn (cheap), injected at
    API-call time so it never pollutes the cached/stored system prompt, and
    wrapped in sentinels so expired copies are stripped from replayed history
    (only THIS turn's copy survives as 'current')."""
    try:
        from datetime import datetime
        now_line = datetime.now().astimezone().strftime("%A, %B %d, %Y %H:%M %Z")
    except Exception:
        now_line = "(unavailable)"

    # Context-isolated agents (black-box tester) must not see the team's project
    # tree or inter-agent chatter — only the time and their own cron schedule.
    agent_cfg = ((full_config or {}).get("agents", {}) or {}).get(agent_id, {})
    if agent_cfg.get("context_isolated"):
        crons = _build_cron_summary(full_config, agent_id)
        return (
            f"{LIVE_CONTEXT_BEGIN}\n"
            "--- LIVE CONTEXT (auto-refreshed each turn; only this latest copy is current) ---\n"
            f"Current time: {now_line}\n\n"
            "Your scheduled wake-ups (manage with schedule_wakeup / cancel_wakeup):\n"
            f"{crons}\n"
            f"{LIVE_CONTEXT_END}"
        )

    brief = _read_workspace_brief(team_id)
    tree = _build_workspace_tree(team_id, max_entries=LIVE_CTX_TREE_MAX_ENTRIES)
    recent = _recent_peer_messages(team_id, full_config, limit=LIVE_CTX_PEER_MESSAGES)
    decisions = _recent_decisions(team_id, limit=LIVE_CTX_DECISIONS)
    delegations = _open_delegations_block(team_id, agent_id)
    completed = _recently_completed_block(team_id, limit=LIVE_CTX_COMPLETED)
    humans = _waiting_on_human_block(team_id, full_config)
    crons = _build_cron_summary(full_config, agent_id)
    return (
        f"{LIVE_CONTEXT_BEGIN}\n"
        "--- LIVE TEAM CONTEXT (auto-refreshed; ONLY this latest copy is current — "
        "older copies in the conversation are expired) ---\n"
        f"Current time: {now_line}\n\n"
        "PROJECT BRIEF (workspace.md — the single source of truth; re-read fresh "
        "each turn, so a teammate's edit shows up here next turn. Edit it with "
        "write_file when the project context changes):\n"
        f"{brief}\n\n"
        "Shared project directory (every teammate works in this one tree — this is "
        "what already exists; read any path here directly):\n"
        f"{tree}\n\n"
        f"DECISION LOG — the team's last {LIVE_CTX_DECISIONS} decisions (oldest "
        "first). This is the shared memory; check it before acting so you don't "
        "redo or contradict settled work. Append with log_decision (one line, "
        "sparingly):\n"
        f"{decisions}\n\n"
        "OUTSTANDING WORK (delegation ledger — close items by sending a RESULT):\n"
        f"{delegations}\n\n"
        "RECENTLY COMPLETED (already delivered — do NOT re-delegate or redo these; "
        "use the existing results):\n"
        f"{completed}\n\n"
        "WAITING ON HUMAN (unanswered ask_human/takeover requests across the team):\n"
        f"{humans}\n\n"
        f"Last {LIVE_CTX_PEER_MESSAGES} messages between teammates "
        "(send_peer_message), oldest first:\n"
        f"{recent}\n\n"
        "Your scheduled cron wake-ups (manage with schedule_wakeup / cancel_wakeup):\n"
        f"{crons}\n"
        f"{LIVE_CONTEXT_END}"
    )


def compose_agent_soul(
    agent_cfg: Dict[str, Any],
    full_config: Optional[Dict[str, Any]] = None,
    include_role: bool = True,
) -> str:
    """Build the full ephemeral system prompt for an agent.

    When include_role is False, the trailing "YOUR ROLE" block is omitted —
    used when the role identity is instead written to SOUL.md (so it is not
    duplicated in both the cached prompt and the ephemeral prompt).
    """
    agent_name = agent_cfg.get("name", "Agent")
    team_id = agent_cfg.get("team_id", "default")
    peers = agent_cfg.get("allowed_peers", [])
    peers_str = ", ".join(peers) if peers else "(none — you cannot message any peers yet)"
    # The shared work surface (all agents collaborate here) + the team metadata
    # dir that holds the brief and changelog. No per-agent workspace folder.
    project_dir = str(_get_project_dir(team_id, full_config))
    team_workspace = str(_get_team_workspace_path(team_id))

    # Build the compact peer roster if we have full config. (We deliberately no
    # longer inline the full-team org diagram + member list into every agent's
    # prompt — that was O(N²) tokens; an agent only needs its own linked peers.)
    peer_roster = "(config not available for peer roster)"
    if full_config:
        peer_roster = _build_peer_roster(full_config, team_id, agent_cfg.get("agent_id", "unknown"))

    # Context-isolated agents (e.g. a black-box QA tester) deliberately get NO
    # product brief, roadmap, or org chart — they must discover the product fresh,
    # as an outside customer would. They keep only the swarm mechanics + their role
    # + the bare list of peers to report findings to.
    isolated = bool(agent_cfg.get("context_isolated"))

    # NOTE: workspace.md (the project brief) is intentionally NOT inlined here.
    # It is injected fresh per-turn via compose_live_context (like the decision
    # log) so an agent's edit is visible to the whole team on the very next turn
    # instead of being frozen into the cached system prompt until a restart.

    # NOTE: the soul template is authored as free-form Markdown and contains
    # literal braces (JSON examples, the lease syntax `agent={your_name}`, etc.),
    # so we CANNOT use str.format() — it would treat those as fields and KeyError.
    # Substitute only our known placeholders explicitly.
    _subs = {
        "{agent_name}": agent_name,
        "{team_id}": team_id,
        "{allowed_peers_list}": peers_str,
        "{sweep_interval}": str(SWEEP_INTERVAL_SECONDS),
        "{project_dir}": project_dir,
        "{team_workspace}": team_workspace,
    }
    common = COMMON_SOUL_TEMPLATE
    for _k, _v in _subs.items():
        common = common.replace(_k, _v)

    if isolated:
        body = (
            f"{common}\n"
            f"--- REPORTING ---\n"
            f"You are not wired into the team's internal context and you do not see the "
            f"org chart. When you find a bug, breakage, or confusing experience, report it "
            f"via send_peer_message to: {peers_str}.\n"
        )
    else:
        body = (
            f"{common}\n"
            f"--- TEAM ORGANIZATION ---\n"
            f"{peer_roster}\n"
        )

    if include_role:
        role = agent_cfg.get("role_soul", f"You are the {agent_name}.")
        body += (
            f"\n{'=' * 60}\n"
            f"YOUR ROLE\n"
            f"{'=' * 60}\n"
            f"{role}\n"
        )

    return body
