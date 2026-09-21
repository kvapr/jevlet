"""How the decision layer sits in a single-turn, gated-write agent (the on-device assistant).

Two hooks, both advisory, neither executes anything:

  pre_turn(request)        -> which tool schemas to load into the small context, plus an optional
                              one-line hint for the system prompt when the request is ambiguous
  annotate_proposal(...)   -> the sentence and probability shown on the confirm card for a
                              proposed call ("you asked to cut PoE power; this shuts the whole port")

The backend is any jevlet Backend; probabilities must have been calibrated on the deployment's
own labeled rows (see bench/analyze.py) before the thresholds below mean anything. Thresholds
are per action: tighter for destructive writes, looser for reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..backends.base import Backend
from ..schema import Choice, Noul, confidence

# the production 4-tool profile and the schemas that fit beside it in a 4K compiled context
CORE = ["query_device_info", "configure_device_setting"]
PROFILES = {
    "diagnose": CORE + ["diagnose_port_issues", "search_documentation"],
    "events": CORE + ["get_network_events", "diagnose_port_issues"],
    "camera": CORE + ["get_camera_detections", "search_documentation"],
    "cable": CORE + ["calibrate_cable_baseline", "diagnose_port_issues"],
    "docs": CORE + ["search_documentation", "diagnose_port_issues"],
    "default": CORE + ["diagnose_port_issues", "search_documentation"],
}
TOOL_TO_PROFILE = {"get_network_events": "events", "get_camera_detections": "camera",
                   "calibrate_cable_baseline": "cable", "search_documentation": "docs",
                   "diagnose_port_issues": "diagnose"}

# semantic class of each catalog write, for the "what you asked vs what this does" comparison
OP_CLASS = {
    "disable_port": "port_admin_down", "enable_port": "port_admin_up",
    "configure_poe_mode": "poe_power_on_off", "configure_poe_power_limit": "poe_limit",
    "configure_access_vlan": "access_vlan", "configure_trunk_allowed_vlans": "trunk_vlans",
    "configure_port_speed": "port_speed", "configure_port_duplex": "port_duplex",
    "set_port_description": "port_description", "create_vlan": "vlan_create",
}
CLASS_TEXT = {
    "port_admin_down": "shuts the interface down (link AND PoE power go down until re-enabled)",
    "port_admin_up": "brings the interface back up",
    "poe_power_on_off": "changes PoE power delivery only; the link stays up",
    "poe_limit": "caps the PoE wattage on the port",
    "access_vlan": "puts the port in one untagged VLAN (access port)",
    "trunk_vlans": "makes the port a trunk with an allowed-VLAN list",
    "port_speed": "changes the port speed", "port_duplex": "changes the duplex setting",
    "port_description": "changes the port label only", "vlan_create": "creates a VLAN",
}


@dataclass
class PreTurn:
    profile: str
    tools: list[str]
    tool_choice: str
    tool_confidence: float
    is_write_p: float
    hint: str | None
    answers: dict = field(default_factory=dict)


def pre_turn(backend: Backend, request: str, tool_options: dict, clarify_below: float = 0.6) -> PreTurn:
    """Route the request to a tool profile and decide whether to ask the LM to state its reading."""
    res = backend.decide(request, {
        "tool": Choice("Which tool should the network-switch assistant use for this operator request?", tool_options),
        "is_write": Noul("Does the request ask to CHANGE a setting on the switch, as opposed to viewing, checking or diagnosing?"),
    })
    tool_a = res.answers["tool"]
    opts = list(tool_options.keys())
    i = max(range(len(opts)), key=lambda j: tool_a.probs[j])
    tool, conf = opts[i], confidence(tool_a.probs)
    p_write = res.answers["is_write"].probs[1]
    profile = TOOL_TO_PROFILE.get(tool, "default") if conf >= clarify_below else "default"
    hint = None
    if p_write > 0.5 and conf < clarify_below:
        top2 = sorted(zip(opts, tool_a.probs), key=lambda x: -x[1])[:2]
        hint = (f"The request may mean either {top2[0][0]} or {top2[1][0]}; state your reading of it "
                f"explicitly in the confirmation message.")
    return PreTurn(profile, PROFILES[profile], tool, conf, p_write, hint,
                   {k: a.probs for k, a in res.answers.items()})


@dataclass
class ProposalCheck:
    matches_p: float
    level: str            # "ok" | "check" | "mismatch"
    text: str


def annotate_proposal(backend: Backend, request: str, command: str, params: dict, desc: str = "",
                      ok_above: float = 0.85, mismatch_below: float = 0.4) -> ProposalCheck:
    """One NOUL on (request, proposed call). Never blocks; the human still clicks."""
    state = (f"REQUEST: {request}\nPROPOSED ACTION: {command} {params}" + (f"\n  ({desc})" if desc else ""))
    res = backend.decide(state, {
        "matches_request": Noul("Does the PROPOSED ACTION do what the operator asked in the REQUEST — the same kind of "
                                "operation, on the same port or setting, with the same value?",
                                true="the proposal is a faithful execution of the request",
                                false="the effect differs, or the port, value or device differs"),
    })
    p = res.answers["matches_request"].probs[1]
    cls = OP_CLASS.get(command)
    what = CLASS_TEXT.get(cls, command)
    if p >= ok_above:
        return ProposalCheck(p, "ok", f"Check: this {what} (matches your request, p={p:.2f}).")
    if p <= mismatch_below:
        return ProposalCheck(p, "mismatch", f"Check: this {what} — that may NOT be what you asked for (p={p:.2f}). "
                                            f"Read the command before approving.")
    return ProposalCheck(p, "check", f"Check: this {what}; the match to your request is uncertain (p={p:.2f}).")
