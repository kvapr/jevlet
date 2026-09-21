"""The netops decision dataset: operator requests to a managed PoE switch, with typed labels.

Sources
  data/raw/test_cases.yaml            65 canonical requests (4 difficulty levels)           [private copy]
  data/raw/test_cases_paraphrase.yaml 195 paraphrases (styles A casual / B verbose / C indirect)
  data/raw/tools_flat.json            the 33-command flat catalog with descriptions
  data/raw/results/*.json             generative tool-calling runs of several models (one-hot calls)
  data/authored/*.yaml                rows authored for this project (adversarial confusables, other intents,
                                      training paraphrases), each verified by an independent pass

`python -m bench.netops build` writes the publishable, sanitized dataset to data/netops/:
  canonical.yaml, paraphrase.yaml, authored/*.yaml, catalog.json, generative_runs/*.json
Sanitization: the real inventory name is replaced by "@sw1"; no addresses or credentials are kept.

Question batteries (what a System One layer is asked per request):
  router:  tool CHOICE (8 incl. none) · command CHOICE (catalog+none) · is_write NOUL · touches_poe_power NOUL
  gate:    matches_request NOUL over (request, proposed call)  — labels from real generative misses + synthetic swaps
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path

import yaml

from jevlet.schema import Choice, Noul, Score

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "netops"
AUTHORED = ROOT / "data" / "authored"

# The private inventory name of the bench device never appears in this repository: the builder
# reads it from the environment (or data/raw/private.json, gitignored) and replaces it everywhere.
PUB_NAME = "sw1"


def _private() -> dict:
    """{"real_device_name": "...", "redact": ["...", ...]} — never committed (data/raw is gitignored)."""
    d = {}
    if (RAW / "private.json").exists():
        d = json.load(open(RAW / "private.json"))
    if os.environ.get("JEVLET_REAL_DEVICE_NAME"):
        d["real_device_name"] = os.environ["JEVLET_REAL_DEVICE_NAME"]
    return d


_PRIV = _private()
REAL_NAME = _PRIV.get("real_device_name", "")
REDACT = [s for s in _PRIV.get("redact", []) if s]
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

WRAPPER_TOOLS = ("query_device_info", "configure_device_setting")
READ_CMDS = ["show_port_status", "show_all_ports_status", "show_poe_status", "show_all_poe_status",
             "show_poe_config", "show_all_poe_config", "show_poe_power_budget", "show_cable_diagnostics",
             "show_all_cable_diagnostics", "show_vlan_config", "show_vlan_membership", "show_system_info",
             "show_ip_interface"]
WRITE_CMDS = ["enable_port", "disable_port", "configure_poe_mode", "configure_port_speed",
              "configure_port_duplex", "configure_access_vlan", "configure_poe_power_limit",
              "set_port_description", "configure_trunk_allowed_vlans", "create_vlan"]
CATALOG_CMDS = READ_CMDS + WRITE_CMDS
NONE = "none_of_these"

TOOL_OPTIONS = {
    "query_device_info": {"what": "Read-only status or monitoring query on the switch: port status, PoE status or config, VLAN tables, system info, cable diagnostics. Changes nothing.",
                          "not_for": "anything that changes a setting; troubleshooting a described fault"},
    "configure_device_setting": {"what": "Any request to CHANGE switch configuration: enable/disable a port link, PoE power on/off or limit, speed, duplex, VLAN assignment, port description. Shows a confirmation before running.",
                                 "not_for": "requests that only ask to see or check something"},
    "diagnose_port_issues": {"what": "Multi-step troubleshooting of ONE port that is down, flapping or misbehaving: status, error counters, PoE, cable test, syslog history for that port.",
                             "not_for": "a plain status read with no symptom described; changes"},
    "search_documentation": {"what": "How-to and syntax questions answered from the switch CLI/API/web documentation.",
                             "not_for": "acting on the switch"},
    "get_network_events": {"what": "Recent switch-wide syslog events by time window and category (link up/down, PoE, auth failures, reboots, config changes).",
                           "not_for": "one port's diagnosis; live status"},
    "get_camera_detections": {"what": "What a connected camera sees now, or its detection history (people, vehicles, objects).",
                              "not_for": "anything about switch ports"},
    "calibrate_cable_baseline": {"what": "Record the healthy cable length of a port as a baseline for later cable-fault localization.",
                                 "not_for": "reading cable diagnostics"},
    NONE: {"what": "Not a single-device switch operation this assistant can perform: chit-chat, out-of-scope, fleet-wide requests, non-requests."},
}


# ----------------------------------------------------------------------------- loading
DROP_KEYS = ("url", "base_url", "endpoint", "host", "server", "address")


def _sanitize_text(s: str) -> str:
    if REAL_NAME:
        s = re.sub(re.escape(REAL_NAME), PUB_NAME, s, flags=re.I)
    for w in REDACT:
        s = re.sub(re.escape(w), "<redacted>", s, flags=re.I)
    return IP_RE.sub("<addr>", s)


def _sanitize_obj(o):
    if isinstance(o, str):
        return _sanitize_text(o)
    if isinstance(o, dict):
        return {k: _sanitize_obj(v) for k, v in o.items() if k not in DROP_KEYS}
    if isinstance(o, list):
        return [_sanitize_obj(v) for v in o]
    return o


def load_levels(path) -> list[dict]:
    d = yaml.safe_load(open(path))
    rows = []
    for level, items in d.items():
        if not isinstance(items, list):
            continue
        for it in items:
            if not isinstance(it, dict) or "input" not in it:
                continue
            r = dict(it)
            r["level"] = level
            rows.append(r)
    return rows


def catalog() -> dict[str, str]:
    """command/tool name -> one-line description (sanitized). Uses the published copy when the
    private raw file is absent (the raw directory is never synced or committed)."""
    if not (RAW / "tools_flat.json").exists() and (OUT / "catalog.json").exists():
        return json.load(open(OUT / "catalog.json"))
    t = json.load(open(RAW / "tools_flat.json"))
    out = {}
    for x in t:
        f = x.get("function", x)
        d = (f.get("description") or "").split("\n")[0]
        d = re.sub(r"\(e\.g\.,? ?'[^)]*\)", "", d)          # drop example device names
        d = _sanitize_text(d).replace("issw", "sw2").replace("switch1", "sw2")
        out[f["name"]] = d.strip()
    return out


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "generative_runs").mkdir(exist_ok=True)
    (OUT / "authored").mkdir(exist_ok=True)
    canon = [_sanitize_obj(r) for r in load_levels(RAW / "test_cases.yaml")]
    para = [_sanitize_obj(r) for r in load_levels(RAW / "test_cases_paraphrase.yaml")]
    for r in canon:
        r["style"] = "canon"
        r["base_input"] = r["input"]
    yaml.safe_dump(canon, open(OUT / "canonical.yaml", "w"), sort_keys=False, allow_unicode=True)
    yaml.safe_dump(para, open(OUT / "paraphrase.yaml", "w"), sort_keys=False, allow_unicode=True)
    json.dump(catalog(), open(OUT / "catalog.json", "w"), indent=1)
    kept = 0
    for p in sorted(glob.glob(str(RAW / "results" / "*.json"))):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if not isinstance(d, dict) or "results" not in d:
            continue
        d = _sanitize_obj(d)
        d.get("summary", {}).pop("url", None)
        json.dump(d, open(OUT / "generative_runs" / os.path.basename(p), "w"))
        kept += 1
    for p in sorted(glob.glob(str(AUTHORED / "*.yaml"))):
        rows = yaml.safe_load(open(p)) or []
        rows = [_sanitize_obj(r) for r in rows]
        yaml.safe_dump(rows, open(OUT / "authored" / os.path.basename(p), "w"), sort_keys=False, allow_unicode=True)
    print(f"wrote {len(canon)} canonical, {len(para)} paraphrase, {kept} generative runs, "
          f"{len(glob.glob(str(OUT / 'authored' / '*.yaml')))} authored files -> {OUT}")


def load_rows(splits=("canonical", "paraphrase", "authored")) -> list[dict]:
    """Rows of the published dataset, each with: input, expected_tool, expected_command, expected_params,
    style, base_input, level/category, and (authored only) is_write/touches_poe_power/ambiguity/... labels."""
    rows = []
    if "canonical" in splits:
        for r in yaml.safe_load(open(OUT / "canonical.yaml")):
            r.setdefault("split", "canonical")
            rows.append(r)
    if "paraphrase" in splits:
        for r in yaml.safe_load(open(OUT / "paraphrase.yaml")):
            r.setdefault("split", "paraphrase")
            rows.append(r)
    if "authored" in splits:
        for p in sorted(glob.glob(str(OUT / "authored" / "*.yaml"))):
            name = os.path.basename(p)[:-5]
            if name.startswith("train_"):
                continue
            for r in yaml.safe_load(open(p)) or []:
                r.setdefault("split", name)
                r.setdefault("base_input", r["input"])
                r.setdefault("expected_params", {})
                rows.append(r)
    for r in rows:
        r.setdefault("expected_command", "")
        r["expected_command"] = r["expected_command"] or ""
        # derived labels where not authored
        if "is_write" not in r:
            r["is_write"] = r["expected_tool"] == "configure_device_setting"
        if "touches_poe_power" not in r:
            c = r["expected_command"]
            r["touches_poe_power"] = c in ("configure_poe_mode", "configure_poe_power_limit", "show_poe_status",
                                           "show_all_poe_status", "show_poe_config", "show_all_poe_config",
                                           "show_poe_power_budget")
        r.setdefault("ambiguity", 0)
        r.setdefault("alt_ok", [])
        r.setdefault("dangerous_pair", "none")
    return rows


# ----------------------------------------------------------------------------- questions
def router_questions(cat: dict[str, str] | None = None, commands: list[str] | None = None) -> dict:
    cat = cat or catalog()
    cmds = commands or CATALOG_CMDS
    return {
        "tool": Choice("Which tool should the network-switch assistant use for this operator request?", TOOL_OPTIONS),
        "command": Choice("If the request maps to one catalog command on the switch, which one? "
                          "Note the confusable pairs: disable_port shuts the interface (link AND power go down) while "
                          "configure_poe_mode only turns PoE power off or on with the link left up; configure_access_vlan "
                          "puts an access port in ONE untagged VLAN while configure_trunk_allowed_vlans makes a trunk with an "
                          "allowed-VLAN list; show_vlan_config lists per-port VLAN settings while show_vlan_membership lists "
                          "each VLAN's member ports.",
                          {**{c: cat.get(c, c) for c in cmds},
                           NONE: "The request is not one catalog command (troubleshooting, documentation, events, cameras, out of scope)."}),
        "is_write": Noul("Does the request ask to CHANGE a setting on the switch, as opposed to viewing, checking or diagnosing?",
                         true="it asks to enable, disable, set, assign, configure, cap, name, move, or otherwise alter something",
                         false="it only asks to show, check, look at, investigate, or explain"),
        "touches_poe_power": Noul("Is the request about PoE power delivery (power on/off, watt limits, power draw, powered devices), "
                                  "rather than about the port's link, traffic, speed, duplex or VLAN?",
                                  true="it talks about power, PoE, feeding or cutting power, wattage, or a powered device losing power",
                                  false="it talks about the link, traffic, speed, duplex, VLAN, or the port generically"),
    }


def ambiguity_question() -> dict:
    return {"ambiguity": Score("How ambiguous is what the operator wants done?",
                               ["exactly one clear action, target and value",
                                "a minor gap (a port or value is missing) but only one sensible reading",
                                "two plausible readings with different effects on the switch",
                                "it is unclear what change, if any, is wanted"])}


def gate_questions() -> dict:
    return {
        "matches_request": Noul("Does the PROPOSED ACTION do what the operator asked in the REQUEST — the same kind of "
                                "operation, on the same port or setting, with the same value?",
                                true="the proposal is a faithful execution of the request",
                                false="the effect differs (e.g. the request was to cut PoE power but the proposal shuts the whole "
                                      "port down; a request to put a port in one VLAN but the proposal makes it a trunk), or the "
                                      "port, value or device differs"),
    }


def gate_state(request: str, command: str, params: dict) -> str:
    desc = catalog().get(command, "")
    return (f"REQUEST: {request}\n"
            f"PROPOSED ACTION: {command} {json.dumps(params, sort_keys=True)}"
            + (f"\n  ({desc})" if desc else ""))


# ----------------------------------------------------------------------------- gate pairs
NEG_SWAP = {
    "disable_port": lambda p: ("configure_poe_mode", {"port": p.get("port"), "mode": "disable"}),
    "enable_port": lambda p: ("configure_poe_mode", {"port": p.get("port"), "mode": "enable"}),
    "configure_poe_mode": lambda p: (("disable_port" if p.get("mode") == "disable" else "enable_port"), {"port": p.get("port")}),
    "configure_access_vlan": lambda p: ("configure_trunk_allowed_vlans", {"port": p.get("port"), "vlans": str(p.get("vlan_id", p.get("vlan", "")))}),
    "configure_port_speed": lambda p: ("disable_port", {"port": p.get("port")}),
    "configure_port_duplex": lambda p: ("configure_port_speed", {"port": p.get("port"), "speed": "100"}),
    "configure_poe_power_limit": lambda p: ("configure_poe_mode", {"port": p.get("port"), "mode": "disable"}),
    "set_port_description": lambda p: ("disable_port", {"port": p.get("port")}),
    "show_port_status": lambda p: ("show_poe_status", {"port": p.get("port")}),
    "show_poe_status": lambda p: ("show_port_status", {"port": p.get("port")}),
    "show_vlan_config": lambda p: ("show_vlan_membership", {}),
    "show_all_ports_status": lambda p: ("show_poe_power_budget", {}),
    "show_system_info": lambda p: ("show_vlan_config", {}),
    "show_poe_power_budget": lambda p: ("show_all_ports_status", {}),
}


def wrong_port(p: dict):
    port = p.get("port")
    if not port or "/" not in str(port):
        return None
    a, b = str(port).split("/")
    q = dict(p)
    q["port"] = f"{a}/{int(b) % 12 + 1}"
    return q


def _call_of(rec: dict):
    """Extract (command, params) from a generative run record (handles wrapper and bare-command shapes)."""
    gt, ga = rec.get("got_tool"), rec.get("got_args") or {}
    if gt in WRAPPER_TOOLS:
        cmd = ga.get("command_name")
        params = ga.get("parameters") or {k: v for k, v in ga.items() if k not in ("device_name", "command_name")}
    else:
        cmd = gt
        params = ga.get("parameters") or {k: v for k, v in ga.items() if k not in ("device_name",)}
    return cmd, params or {}


def gate_pairs(run_file: str | None = None, rows: list[dict] | None = None, synthetic=True) -> list[dict]:
    """Labeled (request, proposed call, label) triples for the confirm-gate check.
    label 1 = the proposal is what the operator asked; 0 = it is not.
    real: the generative model's actual call from a run file (label = its e2e correctness in that run);
    synthetic: gold call (1), swapped operation (0), wrong port (0)."""
    rows = rows or load_rows(("canonical", "paraphrase"))
    by_input = {r["input"]: r for r in rows}
    pairs = []
    if run_file:
        d = json.load(open(run_file))
        for rec in d["results"]:
            r = by_input.get(rec["input"])
            if r is None:
                continue
            cmd, params = _call_of(rec)
            if not cmd:
                continue
            e2e = bool(rec.get("ok_tool") and rec.get("ok_command") and rec.get("ok_params_lenient") and rec.get("ok_device"))
            # the gate label is SEMANTIC: does the proposed call do what was asked? A bare-command
            # format slip with the right command and parameters matches; the e2e grade does not.
            exp_params = r.get("expected_params") or {}
            params_ok = all(str(params.get(k)) == str(v) for k, v in exp_params.items())
            same_cmd = cmd == r["expected_command"]
            label = int(same_cmd and params_ok)
            if e2e:
                miss_kind = "none"
            elif same_cmd and params_ok:
                miss_kind = "format_only"
            elif same_cmd:
                miss_kind = "params"
            elif cmd == "diagnose_port_issues":
                miss_kind = "diagnose_instead"
            else:
                miss_kind = "confusion"
            pairs.append({"input": r["input"], "base_input": r["base_input"], "style": r["style"],
                          "command": cmd, "params": params, "label": label, "kind": "real", "e2e": int(e2e),
                          "miss_kind": miss_kind, "true_command": r["expected_command"],
                          "dangerous": _is_dangerous(r["expected_command"], cmd)})
    if synthetic:
        for r in rows:
            c, p = r["expected_command"], r.get("expected_params") or {}
            if not c:
                continue
            pairs.append({"input": r["input"], "base_input": r["base_input"], "style": r["style"], "command": c,
                          "params": p, "label": 1, "kind": "gold", "true_command": c, "dangerous": False})
            if c in NEG_SWAP:
                nc, np_ = NEG_SWAP[c](p)
                pairs.append({"input": r["input"], "base_input": r["base_input"], "style": r["style"], "command": nc,
                              "params": np_, "label": 0, "kind": "swap_op", "true_command": c, "dangerous": _is_dangerous(c, nc)})
            wp = wrong_port(p)
            if wp:
                pairs.append({"input": r["input"], "base_input": r["base_input"], "style": r["style"], "command": c,
                              "params": wp, "label": 0, "kind": "wrong_port", "true_command": c, "dangerous": False})
    return pairs


DANGEROUS_PAIRS = [{"disable_port", "configure_poe_mode"}, {"enable_port", "configure_poe_mode"},
                   {"configure_access_vlan", "configure_trunk_allowed_vlans"}, {"configure_port_speed", "disable_port"},
                   {"configure_poe_power_limit", "configure_poe_mode"}]


def _is_dangerous(true_cmd: str, proposed: str) -> bool:
    return true_cmd != proposed and any({true_cmd, proposed} == s for s in DANGEROUS_PAIRS)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build()
    else:
        rows = load_rows()
        import collections
        print(len(rows), "rows;", collections.Counter(r["split"] for r in rows))
