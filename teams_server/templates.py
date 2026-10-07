"""Team templates — predefined agent teams and the onboarding draft flow.

Two sources of team specs:

  * PREDEFINED_TEMPLATES — curated, ready-made business teams shown in the
    template gallery ("choose a ready-made team").
  * Draft — the Architect's proposed team saved via the master_save_template
    tool. Stored at DATA_ROOT/onboarding_draft.json so the dashboard can show
    it on a review screen before anything is built.

Both feed instantiate_template(), which turns a spec into a live team:
creates the team, the agents, the bidirectional peer links, the shared
workspace.md brief, and registers the running daemons.

Spec shape:
    {
        "name": "Marketing Team",
        "description": "one-liner shown on the review screen",
        "brief": "workspace.md content — the shared brief",
        "agents": [
            {
                "key": "strategist",            # slug hint; deduped on create
                "display_name": "Strategist",
                "role": "one-line description for the avatar card",
                "role_soul": "the agent's identity prompt",
                "peers": ["copywriter"],         # keys; links are bidirectional
                "is_supervisor": false,
                "autonomous": false,
            }, ...
        ],
    }
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

log = logging.getLogger("teams.templates")

DRAFT_NAME = "onboarding_draft.json"

# ---------------------------------------------------------------------------
# Predefined templates
#
# Same design rules the Architect enforces: exactly one supervisor linked to
# every worker, one autonomous "driver" (the lead), hub-and-spoke topology,
# short second-person souls with non-overlapping mandates.
# ---------------------------------------------------------------------------

def _sup(display_name: str = "Manager") -> Dict[str, Any]:
    return {
        "key": "manager",
        "display_name": display_name,
        "role": "Reviews everyone's work and keeps the team on track",
        "role_soul": "",   # blank → default supervisor brief
        "is_supervisor": True,
        "autonomous": False,
        "peers": [],       # filled in below — linked to every worker
    }


def _with_manager(agents: List[Dict[str, Any]], manager_name: str = "Manager") -> List[Dict[str, Any]]:
    """Append a supervisor linked to every worker in `agents`."""
    sup = _sup(manager_name)
    sup["peers"] = [a["key"] for a in agents]
    return agents + [sup]


PREDEFINED_TEMPLATES: List[Dict[str, Any]] = [
    {
        "id": "marketing",
        "name": "Marketing Team",
        "tagline": "Content, campaigns, and audience growth",
        "outcome": "Steady stream of posts, market intel, and a weekly what-worked report",
        "description": "Researches your market, writes and schedules content, "
                       "and reports what's working.",
        "agents": _with_manager([
            {
                "key": "strategist",
                "display_name": "Strategist",
                "role": "Plans campaigns and directs the team",
                "role_soul": (
                    "You are the marketing STRATEGIST and team lead. You own the "
                    "plan: research the market and competitors, define positioning, "
                    "and break the goal into concrete assignments for your teammates. "
                    "Delegate writing to the Copywriter, numbers to the Analyst, and "
                    "publishing to the Publisher. Drive every campaign to a shipped "
                    "deliverable — never stop at a draft."
                ),
                "peers": ["copywriter", "analyst", "publisher"],
                "is_supervisor": False,
                "autonomous": True,
            },
            {
                "key": "copywriter",
                "display_name": "Copywriter",
                "role": "Writes posts, emails, and ad copy",
                "role_soul": (
                    "You are the COPYWRITER. You write in the brand's voice: social "
                    "posts, email campaigns, blog drafts, and ad copy. Take briefs "
                    "from the Strategist, deliver polished copy, and revise on "
                    "feedback. You write — others research, publish, and measure."
                ),
                "peers": ["strategist", "publisher"],
                "is_supervisor": False,
                "autonomous": False,
            },
            {
                "key": "analyst",
                "display_name": "Analyst",
                "role": "Tracks performance and reports insights",
                "role_soul": (
                    "You are the ANALYST. You measure what the team ships: campaign "
                    "performance, audience growth, and what's working. Produce short, "
                    "honest reports with numbers, and flag opportunities back to the "
                    "Strategist."
                ),
                "peers": ["strategist"],
                "is_supervisor": False,
                "autonomous": False,
            },
            {
                "key": "publisher",
                "display_name": "Publisher",
                "role": "Schedules and publishes content",
                "role_soul": (
                    "You are the PUBLISHER. You take finished content and get it in "
                    "front of people: schedule posts, manage distribution channels, "
                    "and monitor engagement. When a platform needs a login or "
                    "approval you can't do yourself, ask the human."
                ),
                "peers": ["strategist", "copywriter"],
                "is_supervisor": False,
                "autonomous": False,
            },
        ]),
        "brief": (
            "# Marketing Team\n\n"
            "## Goal\nGrow awareness and engagement for the business through "
            "consistent, well-targeted content.\n\n"
            "## How we work\n- Strategist plans and delegates; Copywriter writes; "
            "Publisher distributes; Analyst measures.\n"
            "- Ask the human before publishing anything external.\n"
            "- Log decisions so the whole team stays aligned."
        ),
    },
    {
        "id": "sales",
        "name": "Sales Team",
        "tagline": "Find leads, reach out, and close",
        "outcome": "Qualified leads in your pipeline and personalized outreach drafted",
        "description": "Researches prospects, personalizes outreach, and keeps "
                       "the pipeline moving.",
        "agents": _with_manager([
            {
                "key": "lead",
                "display_name": "Deal Manager",
                "role": "Runs the pipeline and directs the team",
                "role_soul": (
                    "You are the DEAL MANAGER and team lead. You own the pipeline: "
                    "which accounts matter, what stage each deal is in, and what "
                    "happens next. Send prospecting work to the Prospector and "
                    "outreach to Outreach; keep every opportunity moving toward a "
                    "close and report the forecast honestly."
                ),
                "peers": ["prospector", "outreach", "analyst"],
                "is_supervisor": False,
                "autonomous": True,
            },
            {
                "key": "prospector",
                "display_name": "Prospector",
                "role": "Finds and researches leads",
                "role_soul": (
                    "You are the PROSPECTOR. You find and research target accounts "
                    "and leads: fit, signals, contacts, and angles. Deliver tidy, "
                    "sourced research the team can act on — you find them, others "
                    "reach out."
                ),
                "peers": ["lead", "outreach"],
                "is_supervisor": False,
                "autonomous": False,
            },
            {
                "key": "outreach",
                "display_name": "Outreach",
                "role": "Writes personalized outreach and follow-ups",
                "role_soul": (
                    "You are OUTREACH. You turn research into messages: personalized "
                    "emails, follow-ups, and call scripts. Never send anything "
                    "external without human approval — draft it, log it, and ask."
                ),
                "peers": ["lead", "prospector"],
                "is_supervisor": False,
                "autonomous": False,
            },
            {
                "key": "analyst",
                "display_name": "Analyst",
                "role": "Tracks pipeline health and conversion",
                "role_soul": (
                    "You are the ANALYST. You track pipeline metrics, conversion "
                    "rates, and response rates. Produce weekly summaries and flag "
                    "risks to the Deal Manager before they become losses."
                ),
                "peers": ["lead"],
                "is_supervisor": False,
                "autonomous": False,
            },
        ]),
        "brief": (
            "# Sales Team\n\n"
            "## Goal\nBuild a healthy pipeline of qualified opportunities and move "
            "them to close.\n\n"
            "## How we work\n- Deal Manager owns the pipeline and delegates.\n"
            "- Nothing goes to a prospect without human approval.\n"
            "- Log decisions; keep the brief updated with key accounts."
        ),
    },
    {
        "id": "support",
        "name": "Support Team",
        "tagline": "Answer customers and build the knowledge base",
        "outcome": "Faster answers for customers and a knowledge base that grows itself",
        "description": "Triages incoming questions, resolves issues, and turns "
                       "answers into docs.",
        "agents": _with_manager([
            {
                "key": "triage",
                "display_name": "Triage Lead",
                "role": "Sorts incoming issues and assigns work",
                "role_soul": (
                    "You are the TRIAGE LEAD and team coordinator. You watch the "
                    "queue: categorize each issue, set priority, and hand it to the "
                    "right teammate. Keep response times honest and escalate "
                    "anything you can't route."
                ),
                "peers": ["resolver", "documenter", "analyst"],
                "is_supervisor": False,
                "autonomous": True,
            },
            {
                "key": "resolver",
                "display_name": "Resolver",
                "role": "Investigates and answers customer issues",
                "role_soul": (
                    "You are the RESOLVER. You dig into assigned issues, find the "
                    "answer, and draft clear, friendly replies. When an issue needs "
                    "a decision or access you don't have, ask the human — never "
                    "guess at customer-facing answers."
                ),
                "peers": ["triage", "documenter"],
                "is_supervisor": False,
                "autonomous": False,
            },
            {
                "key": "documenter",
                "display_name": "Documenter",
                "role": "Turns resolved issues into help articles",
                "role_soul": (
                    "You are the DOCUMENTER. You turn resolved issues into "
                    "knowledge-base articles and FAQ entries so the same question "
                    "never needs answering twice. Keep the docs accurate and "
                    "up to date."
                ),
                "peers": ["triage", "resolver"],
                "is_supervisor": False,
                "autonomous": False,
            },
            {
                "key": "analyst",
                "display_name": "Analyst",
                "role": "Tracks response times and trends",
                "role_soul": (
                    "You are the ANALYST. You track support metrics — response "
                    "time, resolution rate, recurring issues — and surface trends "
                    "to the Triage Lead so the team improves every week."
                ),
                "peers": ["triage"],
                "is_supervisor": False,
                "autonomous": False,
            },
        ]),
        "brief": (
            "# Support Team\n\n"
            "## Goal\nEvery customer question gets a fast, accurate, friendly "
            "answer — and the answer becomes reusable docs.\n\n"
            "## How we work\n- Triage routes, Resolver answers, Documenter "
            "publishes, Analyst measures.\n"
            "- Escalate to the human when unsure; never invent policy."
        ),
    },
    {
        "id": "operations",
        "name": "Operations Team",
        "tagline": "Monitoring, reporting, and routine work",
        "outcome": "Daily briefs on what happened and early warnings on problems",
        "description": "Watches your workflows, sends daily briefs, and flags "
                       "problems early.",
        "agents": _with_manager([
            {
                "key": "orchestrator",
                "display_name": "Orchestrator",
                "role": "Runs recurring workflows and keeps things moving",
                "role_soul": (
                    "You are the ORCHESTRATOR and team lead. You keep operations "
                    "running: monitor the workflows in scope, restart or escalate "
                    "what stalls, and assign reporting and checks to teammates. "
                    "Reliability over cleverness."
                ),
                "peers": ["reporter", "compliance"],
                "is_supervisor": False,
                "autonomous": True,
            },
            {
                "key": "reporter",
                "display_name": "Reporter",
                "role": "Sends daily summaries and status briefs",
                "role_soul": (
                    "You are the REPORTER. You produce the daily digest: what "
                    "happened, what's blocked, what needs the human's attention. "
                    "Short, scannable, honest — the human should grasp the day in "
                    "thirty seconds."
                ),
                "peers": ["orchestrator"],
                "is_supervisor": False,
                "autonomous": False,
            },
            {
                "key": "compliance",
                "display_name": "Compliance",
                "role": "Checks processes and flags risks",
                "role_soul": (
                    "You are COMPLIANCE. You check that work follows the team's "
                    "rules and flag risks early: policy gaps, missing approvals, "
                    "audit gaps. You don't fix things — you make sure they're "
                    "seen."
                ),
                "peers": ["orchestrator"],
                "is_supervisor": False,
                "autonomous": False,
            },
        ]),
        "brief": (
            "# Operations Team\n\n"
            "## Goal\nKeep the business's routine operations visible, reliable, "
            "and reported.\n\n"
            "## How we work\n- Orchestrator monitors and assigns; Reporter briefs "
            "the human daily; Compliance flags risk.\n"
            "- Escalate anything ambiguous instead of guessing."
        ),
    },
]


# ---------------------------------------------------------------------------
# Spec helpers
# ---------------------------------------------------------------------------

def _slugify(value: str, fallback: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", (value or "").strip().lower()).strip("-_")
    return (slug or fallback)[:64]


def _unique_slug(base: str, taken: set) -> str:
    slug, i = base, 2
    while slug in taken:
        slug = f"{base}-{i}"
        i += 1
    taken.add(slug)
    return slug


def validate_spec(spec: Dict[str, Any]) -> List[str]:
    """Return a list of human-readable problems (empty = OK)."""
    problems: List[str] = []
    if not isinstance(spec, dict):
        return ["spec must be an object"]
    if not (spec.get("name") or "").strip():
        problems.append("spec.name is required")
    agents = spec.get("agents")
    if not isinstance(agents, list) or not agents:
        problems.append("spec.agents must be a non-empty list")
        return problems
    keys = set()
    for i, a in enumerate(agents):
        if not isinstance(a, dict):
            problems.append(f"agents[{i}] must be an object")
            continue
        k = (a.get("key") or a.get("display_name") or "").strip()
        if not k:
            problems.append(f"agents[{i}] needs a key or display_name")
            continue
        keys.add(k)
    for i, a in enumerate(agents):
        for p in (a.get("peers") or []):
            if p not in keys:
                problems.append(f"agents[{i}] links to unknown peer '{p}'")
    return problems


def public_template(t: Dict[str, Any]) -> Dict[str, Any]:
    """The gallery card view of a template (no souls, no wiring detail)."""
    return {
        "id": t["id"],
        "name": t["name"],
        "tagline": t.get("tagline", ""),
        "outcome": t.get("outcome", ""),
        "description": t.get("description", ""),
        "agents": [
            {
                "display_name": a["display_name"],
                "role": a.get("role", ""),
                "is_supervisor": bool(a.get("is_supervisor")),
                "autonomous": bool(a.get("autonomous")),
            }
            for a in t.get("agents", [])
        ],
    }


def get_template(template_id: str) -> Optional[Dict[str, Any]]:
    for t in PREDEFINED_TEMPLATES:
        if t["id"] == template_id:
            return t
    return None


def list_templates() -> List[Dict[str, Any]]:
    return [public_template(t) for t in PREDEFINED_TEMPLATES]


def full_template(template_id: str) -> Optional[Dict[str, Any]]:
    """The complete spec the review screen / instantiate endpoint consume."""
    t = get_template(template_id)
    if not t:
        return None
    spec = {k: v for k, v in t.items() if k != "id"}
    spec["agents"] = [dict(a) for a in t["agents"]]
    return spec


# ---------------------------------------------------------------------------
# Draft (Architect-produced) template
# ---------------------------------------------------------------------------

def _draft_path():
    from teams_server.config import DATA_ROOT

    return DATA_ROOT / DRAFT_NAME


def save_draft(spec: Dict[str, Any]) -> Dict[str, Any]:
    problems = validate_spec(spec)
    if problems:
        raise ValueError("; ".join(problems))
    doc = {
        "spec": spec,
        "saved_at": time.time(),
        "source": "architect",
    }
    path = _draft_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    return doc


def get_draft() -> Optional[Dict[str, Any]]:
    path = _draft_path()
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        log.warning("[templates] failed to read draft: %s", e)
    return None


def clear_draft() -> None:
    try:
        _draft_path().unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Instantiate: spec → live team
# ---------------------------------------------------------------------------

def instantiate_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Build a live team from a template spec.

    Creates the team, every agent (workers first, supervisors last so peers
    resolve), bidirectional links, and the shared workspace.md brief. Returns
    {team_id, team_name, agents:[{agent_name, display_name, ...}]}.

    Daemon registration is the caller's job (it needs the event loop): for each
    returned agent, call register_agent_daemon with the stored cfg.
    """
    from teams_server.config import (
        load_agents_config, create_team, create_agent, save_agent_config,
        set_agent_peers, _get_team_workspace_path,
    )

    problems = validate_spec(spec)
    if problems:
        raise ValueError("; ".join(problems))

    cfg = load_agents_config()
    team_id = _unique_slug(_slugify(spec["name"], "team"), set(cfg["teams"]))
    create_team(cfg, team_id, spec["name"].strip())

    # Workers first, supervisors last — peer links must point at existing agents.
    agents = list(spec["agents"])
    workers = [a for a in agents if not a.get("is_supervisor")]
    supers = [a for a in agents if a.get("is_supervisor")]
    ordered = workers + supers

    taken = set(load_agents_config()["agents"].keys())
    key_to_name: Dict[str, str] = {}
    created: List[Dict[str, Any]] = []

    for a in ordered:
        key = (a.get("key") or a.get("display_name")).strip()
        agent_name = _unique_slug(_slugify(key, "agent"), taken)
        key_to_name[key] = agent_name
        cfg = load_agents_config()
        agent_cfg = create_agent(
            cfg,
            name=agent_name,
            team_id=team_id,
            display_name=(a.get("display_name") or key).strip(),
            allowed_peers=None,   # links set below, after all agents exist
            role_soul=(a.get("role_soul") or "").strip() or None,
            is_supervisor=bool(a.get("is_supervisor")),
        )
        if a.get("autonomous"):
            cfg = load_agents_config()
            cfg["agents"][agent_name]["autonomous"] = True
            save_agent_config(agent_name, cfg["agents"][agent_name])
            agent_cfg = cfg["agents"][agent_name]
        created.append({
            "agent_name": agent_name,
            "display_name": agent_cfg.get("name"),
            "role": a.get("role", ""),
            "is_supervisor": bool(a.get("is_supervisor")),
            "autonomous": bool(a.get("autonomous")),
            "cfg": agent_cfg,
        })

    # Wire bidirectional links (same-team enforced by set_agent_peers).
    for a in ordered:
        key = (a.get("key") or a.get("display_name")).strip()
        peers = [key_to_name[p] for p in (a.get("peers") or []) if p in key_to_name]
        if peers:
            cfg = load_agents_config()
            try:
                set_agent_peers(cfg, key_to_name[key], peers)
            except ValueError as e:
                log.warning("[templates] link wiring for '%s' skipped: %s", key, e)

    # Shared brief — workspace.md injected into every agent's context.
    brief = (spec.get("brief") or "").strip()
    if brief:
        ws_md = _get_team_workspace_path(team_id) / "workspace.md"
        ws_md.parent.mkdir(parents=True, exist_ok=True)
        ws_md.write_text(brief, encoding="utf-8")

    return {
        "team_id": team_id,
        "team_name": spec["name"].strip(),
        "agents": created,
    }
