"""Receipt for the data-hygiene companion page: every number on that page, recomputed.

    python -m bench.hygiene_audit            -> results/hygiene-audit-<date>.json

Read-only with respect to the repository: it counts rows, compares raw and published run files by KEY
NAME only, reads the review log and the git metadata, runs bench/scan_private.sh in its four modes
against this repository and against a throw-away repository seeded with fake names. No private value
is ever written to the receipt: names are counted, not copied.
"""

from __future__ import annotations

import datetime as dt
import glob
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PUB = ROOT / "data" / "netops"
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def sh(cmd, cwd=ROOT, stdin=None):
    p = subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str), capture_output=True, text=True, input=stdin)
    return p.returncode, p.stdout, p.stderr


def keyset(o, prefix="", depth=0, acc=None):
    acc = acc if acc is not None else set()
    if isinstance(o, dict):
        for k, v in o.items():
            acc.add(prefix + k)
            if depth < 2:
                keyset(v, prefix + k + ".", depth + 1, acc)
    elif isinstance(o, list) and o:
        keyset(o[0], prefix + "[].", depth + 1, acc)
    return acc


def dataset_counts():
    out = {}
    for p in ["canonical.yaml", "paraphrase.yaml"] + sorted(f"authored/{os.path.basename(x)}" for x in glob.glob(str(PUB / "authored" / "*.yaml"))):
        out[p] = len(yaml.safe_load(open(PUB / p)) or [])
    out["catalog.json"] = len(json.load(open(PUB / "catalog.json")))
    out["generative_runs"] = len(glob.glob(str(PUB / "generative_runs" / "*.json")))
    out["evaluation_rows"] = out["canonical.yaml"] + out["paraphrase.yaml"] + sum(
        v for k, v in out.items() if k.startswith("authored/") and "train_" not in k)
    rows = total = 0
    for p in ["canonical.yaml", "paraphrase.yaml"] + [f"authored/{os.path.basename(x)}" for x in glob.glob(str(PUB / "authored" / "*.yaml"))]:
        for r in yaml.safe_load(open(PUB / p)) or []:
            rows += 1
            total += 1 if "@sw1" in r.get("input", "") else 0
    out["published_rows_total"] = rows
    out["published_rows_naming_sw1"] = total
    out["generative_run_records_naming_sw1"] = sum(
        sum(1 for r in (json.load(open(f)).get("results") or []) if "@sw1" in json.dumps(r))
        for f in glob.glob(str(PUB / "generative_runs" / "*.json")))
    cred = re.compile(r"\b(password|passwd|secret|api[_-]?key|authorization|bearer|credential)\b", re.I)
    out["published_rows_mentioning_a_credential_word"] = sum(
        1 for p in ["canonical.yaml", "paraphrase.yaml"] + [f"authored/{os.path.basename(x)}" for x in glob.glob(str(PUB / "authored" / "*.yaml"))]
        for r in (yaml.safe_load(open(PUB / p)) or []) if cred.search(r.get("input", "")))
    if (RAW / "tools_flat.json").exists():
        t = json.load(open(RAW / "tools_flat.json"))
        out["raw_catalog_entries_with_example_parenthetical"] = sum(
            1 for x in t if re.search(r"\(e\.g\.,? ?'[^)]*\)", ((x.get("function", x)).get("description") or "").split("\n")[0]))
    return out


def markers():
    n = {"<addr>": 0, "<redacted>": 0}
    for p in list(glob.glob(str(PUB / "**" / "*"), recursive=True)) + [str(ROOT / "docs" / "data-review-log.md")]:
        if os.path.isfile(p):
            t = open(p, errors="replace").read()
            for m in n:
                n[m] += t.count(m)
    return n


def run_files():
    raw = sorted(glob.glob(str(RAW / "results" / "*.json")))
    res = {"raw_files": len(raw), "published_files": len(glob.glob(str(PUB / "generative_runs" / "*.json"))),
           "skipped_no_results_key": 0, "unparsable": 0, "compared": 0, "keys_dropped": set(), "provenance_keys_kept": set()}
    if not raw:
        res["note"] = "data/raw absent on this machine; raw side not compared"
    for p in raw:
        try:
            d = json.load(open(p))
        except Exception:
            res["unparsable"] += 1
            continue
        if not isinstance(d, dict) or "results" not in d:
            res["skipped_no_results_key"] += 1
            kind = "soak run" if "soak" in os.path.basename(p) else "phase probe"
            res.setdefault("skipped_kinds", {})[kind] = res.setdefault("skipped_kinds", {}).get(kind, 0) + 1
            continue
        q = PUB / "generative_runs" / os.path.basename(p)
        if not q.exists():
            continue
        e = json.load(open(q))
        res["compared"] += 1
        res["keys_dropped"] |= keyset(d) - keyset(e)
        res["provenance_keys_kept"] |= set((e.get("provenance") or {}).keys())
    res["keys_dropped"] = sorted(res["keys_dropped"])
    res["provenance_keys_kept"] = sorted(res["provenance_keys_kept"])
    return res


def review_log():
    lines = open(ROOT / "docs" / "data-review-log.md").read().splitlines()
    sections, cur = {}, None
    for ln in lines:
        if ln.startswith("## "):
            cur = ln[3:]
            sections[cur] = {"entries": 0, "dropped_or_removed": 0}
        elif ln.startswith("- ") and cur:
            sections[cur]["entries"] += 1
            # one bullet per removed row, in the four phrasings the two verifiers used; the per-section
            # "- dropped: ..." summary lines repeat those bullets and are not counted
            if not ln.startswith("- dropped:") and re.search(
                    r"\b(DROPPED|REMOVED)\b|\bRemoved\.|dropped #\d+|^- \d+: dropped\b", ln):
                sections[cur]["dropped_or_removed"] += 1
    fingerprint = [i + 1 for i, ln in enumerate(lines) if re.search(r"no '?@sw1'? or other inventory names", ln)]
    return {"sections": sections, "entries_total": sum(s["entries"] for s in sections.values()),
            "dropped_or_removed_total": sum(s["dropped_or_removed"] for s in sections.values()),
            "ambiguity_changes": sum(1 for ln in lines if re.search(r"ambiguity (?:raised |lowered )?\d ?-> ?\d", ln)),
            "evaluation_rows_removed": sum(v["dropped_or_removed"] for k, v in sections.items() if "train" not in k),
            "training_rows_removed": sum(v["dropped_or_removed"] for k, v in sections.items() if "train" in k),
            "sanitizer_fingerprint_lines": fingerprint,
            "abs_path_lines": [i + 1 for i, ln in enumerate(lines) if "/home/" in ln]}


def git_shape():
    _, files, _ = sh(["git", "ls-files"])
    _, n, _ = sh(["git", "rev-list", "--count", "HEAD"])
    _, root, _ = sh(["git", "rev-list", "--max-parents=0", "HEAD"])
    _, stat, _ = sh(["git", "show", "--stat", "--format=", root.strip()])
    m = re.search(r"(\d+) files changed, (\d+) insertions", stat)
    _, reflog, _ = sh(["git", "reflog", "--date=iso", "--format=%h %gd %gs"])
    events = {}
    for ln in reflog.splitlines():
        mm = re.match(r"(\w+) HEAD@\{([^}]+)\} (.*)", ln)
        if not mm:
            continue
        h, when, what = mm.groups()
        if what.startswith("filter-branch"):
            events["last_commit_before_first_push"] = when   # reflog stamp = the rewritten commit's own date;
                                                             # the push time comes from the transcript
        if what.startswith("commit (initial)"):
            events.setdefault("initial_commits", []).append(when)
        if what.startswith("Branch: renamed refs/heads/clean-main"):
            events["clean_history_started"] = when
    _, unreachable, _ = sh("git fsck --unreachable --no-reflogs 2>/dev/null | grep -c commit")
    _, loop_files, _ = sh(r"git grep -l -E '\b([0-9]{1,3}\.){3}[0-9]{1,3}\b' | wc -l")
    _, loop_hits, _ = sh(r"git grep -h -o -E '\b([0-9]{1,3}\.){3}[0-9]{1,3}\b' | sort | uniq -c")
    ip_kinds = {}
    for ln in loop_hits.splitlines():
        c, ip = ln.split()
        ip_kinds["loopback" if ip in ("127.0.0.1", "0.0.0.0") else "other"] = ip_kinds.get("loopback" if ip in ("127.0.0.1", "0.0.0.0") else "other", 0) + int(c)
    _, home_files, _ = sh("git grep -l '/home/[a-z]' | wc -l")
    _, home_lines, _ = sh("git grep -c '/home/[a-z]' | awk -F: '{s+=$2} END{print s+0}'")
    old = events.get("initial_commits", [])
    _, backup_refs, _ = sh(["git", "for-each-ref", "--format=%(refname)", "refs/original/"])
    return {"tracked_files": len(files.split()), "commits": int(n), "root_commit": root.strip()[:7],
            "root_commit_files": int(m.group(1)) if m else None, "root_commit_insertions": int(m.group(2)) if m else None,
            "reflog_events": events, "old_history_initial_commit": old[-1] if len(old) > 1 else None,
            "unreachable_commits_in_local_store": int(unreachable.strip() or 0),
            "old_commits_named_in_head_reflog": sum(1 for ln in reflog.splitlines() if ln[:7] in
                                                    ("04e5192", "3567152", "2410f33", "a5c9bc5", "d2a5a46", "89bce01")),
            "filter_branch_backup_refs_present": len(backup_refs.split()),
            "ipv4_literals": {"files": int(loop_files.strip()), **ip_kinds},
            "abs_home_paths": {"files": int(home_files.strip()), "lines": int(home_lines.strip())}}


def pushes_from_transcript():
    """Push times on the day of the incident, from the session transcript named by $JEVLET_TRANSCRIPT.
    Only timestamps and kinds are recorded; no command text or content is copied."""
    path = os.environ.get("JEVLET_TRANSCRIPT", "")
    if not path or not os.path.exists(path):
        return {"note": "JEVLET_TRANSCRIPT not set or not found; push times not extracted"}
    tz = dt.timezone(dt.timedelta(hours=-5))
    out = []
    for ln in open(path, errors="replace"):
        if "git push" not in ln and "git ls-files" not in ln:
            continue
        try:
            d = json.loads(ln)
        except Exception:
            continue
        ts, msg = d.get("timestamp", ""), d.get("message") or {}
        if not ts.startswith("2026-09-21") or not isinstance(msg, dict):
            continue
        for c in msg.get("content") or []:
            if not (isinstance(c, dict) and c.get("type") == "tool_use" and c.get("name") == "Bash"):
                continue
            cmd = (c.get("input") or {}).get("command", "")
            local = dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(tz)
            kinds = []
            if "git ls-files" in cmd and "grep -l" in cmd:
                kinds.append("whole-tree grep sweep")
            if "git push" in cmd:
                kinds.append("force push" if "--force" in cmd else "push")
            if kinds and local.hour < 16:
                out.append({"local_time": local.strftime("%H:%M:%S"), "kinds": kinds})
    pushes = [e["local_time"] for e in out if "push" in e["kinds"] or "force push" in e["kinds"]]
    first = next((e for e in out if "push" in e["kinds"] or "force push" in e["kinds"]), None)
    clean = next((e for e in out if "force push" in e["kinds"] and e["local_time"] > "15:00"), None)
    res = {"events": out, "pushes_before_clean_history": [t for t in pushes if clean and t < clean["local_time"]]}
    if first and clean:
        f = dt.datetime.strptime(first["local_time"], "%H:%M:%S")
        cl = dt.datetime.strptime(clean["local_time"], "%H:%M:%S")
        fc = dt.datetime.strptime("13:43:31", "%H:%M:%S")
        res["first_push"] = first["local_time"]
        res["clean_history_force_push"] = clean["local_time"]
        res["minutes_first_commit_to_first_push"] = round((f - fc).total_seconds() / 60, 1)
        res["minutes_first_push_to_clean_history"] = round((cl - f).total_seconds() / 60, 1)
    return res


def hook_runs(reps=3):
    """The hook in its four modes against this repository; wall time is the median of `reps` runs."""
    out = {}
    _, head, _ = sh(["git", "rev-parse", "HEAD"])
    push_line = f"refs/heads/main {head.strip()} refs/heads/main {'0'*40}\n"
    for name, args, stdin in [("tree", [], None), ("staged", ["--staged"], None),
                              ("history", ["--history"], None), ("push", ["--push"], push_line)]:
        walls, rc, se = [], None, ""
        for _ in range(reps):
            t = time.time()
            rc, _so, se = sh(["bench/scan_private.sh"] + args, stdin=stdin)
            walls.append(time.time() - t)
        out[name] = {"exit": rc, "wall_s_median": round(sorted(walls)[len(walls) // 2], 2),
                     "wall_s_runs": [round(w, 2) for w in walls],
                     "summary": se.strip().splitlines()[-1]}
    return out


def synthetic():
    """Throw-away repo with FAKE names: the hook must fire in every mode and pass after an orphan squash."""
    d = tempfile.mkdtemp(prefix="hooktest-")
    g = lambda *a, **k: sh(list(a), cwd=d, **k)
    g("git", "init", "-q", "-b", "main", ".")
    g("git", "config", "user.email", "t@example.invalid"); g("git", "config", "user.name", "t")
    os.makedirs(f"{d}/bench"); os.makedirs(f"{d}/data/raw")
    hook = open(ROOT / "bench" / "scan_private.sh").read()
    open(f"{d}/bench/scan_private.sh", "w").write(hook); os.chmod(f"{d}/bench/scan_private.sh", 0o755)
    open(f"{d}/.gitignore", "w").write("data/raw/\n")
    json.dump({"real_device_name": "fakeswitch-zz9", "redact": ["fakehost-qq7"]}, open(f"{d}/data/raw/private.json", "w"))
    open(f"{d}/clean.txt", "w").write("clean row about @sw1 port 3\nserver at http://127.0.0.1:18999\n")
    g("git", "add", "-A"); g("git", "commit", "-qm", "clean")
    r = {}
    r["clean_tree"] = g("bench/scan_private.sh")[0]
    r["clean_history"] = g("bench/scan_private.sh", "--history")[0]
    open(f"{d}/leak_name.txt", "w").write("operator said: FakeSwitch-ZZ9 kill power on 3\n")
    fake_addr = ".".join(["10", "20", "30", "40"])       # assembled at runtime: no address literal in this file
    json.dump({"provenance": {"host": "fakehost-qq7"}, "summary": {"url": f"http://{fake_addr}:8080/v1"}},
              open(f"{d}/leak_run.json", "w"))
    open(f"{d}/leak_blob.bin", "wb").write(b"\x00\x01binary fakeswitch-zz9\n")   # binary: git grep -I skips it
    g("git", "add", "-A")
    rc, so, _ = g("bench/scan_private.sh", "--staged")
    locs = [l.split(" ")[0] for l in so.strip().splitlines() if l]
    r["leak_staged"] = {"exit": rc, "hits": len(locs),
                        "text_files_flagged": sorted({l.split(":")[0] for l in locs}),
                        "binary_file_flagged": any("leak_blob.bin" in l for l in locs)}
    g("git", "commit", "-qm", "leak")
    r["leak_tree"] = g("bench/scan_private.sh")[0]
    r["leak_history"] = g("bench/scan_private.sh", "--history")[0]
    _, head, _ = g("git", "rev-parse", "HEAD")
    r["leak_push"] = g("bench/scan_private.sh", "--push", stdin=f"refs/heads/main {head.strip()} refs/heads/main {'0'*40}\n")[0]
    g("git", "rm", "-q", "leak_name.txt", "leak_run.json", "leak_blob.bin"); g("git", "commit", "-qm", "remove")
    r["fixed_tree_only__tree"] = g("bench/scan_private.sh")[0]
    r["fixed_tree_only__history"] = g("bench/scan_private.sh", "--history")[0]
    _, old_head, _ = g("git", "rev-parse", "HEAD")
    g("git", "checkout", "-q", "--orphan", "clean-main"); g("git", "add", "-A"); g("git", "commit", "-qm", "single clean commit")
    g("git", "branch", "-D", "main", "-q"); g("git", "branch", "-m", "main")
    r["after_orphan_squash__tree"] = g("bench/scan_private.sh")[0]
    r["after_orphan_squash__history"] = g("bench/scan_private.sh", "--history")[0]
    # what filter-branch leaves behind: a backup ref that still reaches the old commits
    g("git", "update-ref", "refs/original/refs/heads/main", old_head.strip())
    r["leftover_backup_ref__history"] = g("bench/scan_private.sh", "--history")[0]
    g("git", "update-ref", "-d", "refs/original/refs/heads/main")
    r["after_deleting_backup_ref__history"] = g("bench/scan_private.sh", "--history")[0]
    return r


def private_census():
    """How much private material the sanitizer removes, counted per key path. Counts only, never values."""
    if not (RAW / "private.json").exists():
        return {"note": "data/raw/private.json absent; census skipped"}
    d = json.load(open(RAW / "private.json"))
    name = str(d.get("real_device_name", ""))
    redact = [str(x) for x in d.get("redact", []) if str(x).strip()]
    out = {"raw": {}, "published": {}, "raw_address_key_paths": [], "raw_redact_key_paths": []}

    def count(text):
        n_name = len(re.findall(re.escape(name), text, re.I)) if name else 0
        ips = IP_RE.findall(text)
        return {"device_name": n_name,
                "ipv4_loopback": sum(1 for i in ips if i in ("127.0.0.1", "0.0.0.0")),
                "ipv4_other": sum(1 for i in ips if i not in ("127.0.0.1", "0.0.0.0")),
                "redact_list": sum(len(re.findall(re.escape(w), text, re.I)) for w in redact)}

    def add(acc, c):
        for k, v in c.items():
            acc[k] = acc.get(k, 0) + v

    for label, paths in [("seed_yaml", [RAW / "test_cases.yaml", RAW / "test_cases_paraphrase.yaml"]),
                         ("generative_runs", sorted(Path(p) for p in glob.glob(str(RAW / "results" / "*.json"))))]:
        acc = {}
        for p in paths:
            if p.exists():
                add(acc, count(open(p, errors="replace").read()))
        out["raw"][label] = acc
    acc = {}
    for p in glob.glob(str(PUB / "**" / "*"), recursive=True):
        if os.path.isfile(p):
            add(acc, count(open(p, errors="replace").read()))
    out["published"]["data/netops"] = acc

    paths_ip, paths_red = set(), set()

    def walk(o, path=""):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, (path + "." + k) if path else k)
        elif isinstance(o, list):
            for v in o:
                walk(v, path + "[]")
        elif isinstance(o, str):
            if any(i not in ("127.0.0.1", "0.0.0.0") for i in IP_RE.findall(o)):
                paths_ip.add(path)
            if any(w.lower() in o.lower() for w in redact):
                paths_red.add(path)
    for p in glob.glob(str(RAW / "results" / "*.json")):
        try:
            walk(json.load(open(p)))
        except Exception:
            pass
    out["raw_address_key_paths"] = sorted(paths_ip)
    out["raw_redact_key_paths"] = sorted(paths_red)
    n_url = n_host = 0
    for p in glob.glob(str(RAW / "results" / "*.json")):
        try:
            dd = json.load(open(p))
        except Exception:
            continue
        if isinstance(dd, dict) and ("url" in dd or "url" in (dd.get("summary") or {})):
            n_url += 1
        if isinstance(dd, dict) and "host" in (dd.get("provenance") or {}):
            n_host += 1
    out["raw_files_with_url_key"] = n_url
    out["raw_files_with_provenance_host"] = n_host
    return out


def incident():
    """What the first push carried, by file and line count, per revision. Names come from the untracked
    file and are never written to the receipt; only counts and file names are."""
    if not (RAW / "private.json").exists():
        return {"note": "data/raw/private.json absent; incident census skipped"}
    d = json.load(open(RAW / "private.json"))
    groups = {"device_name": [str(d.get("real_device_name", ""))],
              "host_name_from_redact_list": [str(x) for x in d.get("redact", [])]}
    revs = {}
    _, reflog, _ = sh(["git", "reflog", "--format=%h %gs"])
    pushed = None
    for ln in reflog.splitlines():                       # the commit that was live when history was rewritten
        if "filter-branch" in ln:
            pushed = ln.split()[0]
    for label, names in groups.items():
        names = [n for n in names if len(n) >= 3]
        if not names:
            continue
        with tempfile.NamedTemporaryFile("w", suffix=".pat", delete=False) as f:
            f.write("\n".join(names) + "\n")
            pat = f.name
        for rev in [r for r in [pushed, "HEAD"] if r]:
            rc, so, _ = sh(["git", "grep", "-I", "-i", "-c", "-F", "-f", pat, rev, "--", "."])
            rows = [l.split(":", 2) for l in so.splitlines() if l.count(":") >= 2]
            revs.setdefault(rev, {})[label] = {"files": len(rows), "lines": sum(int(r[2]) for r in rows),
                                               "file_names": sorted(r[1] for r in rows),
                                               "lines_by_file": {r[1]: int(r[2]) for r in sorted(rows, key=lambda x: x[1])}}
            _, tree, _ = sh(["git", "ls-tree", "-r", "--name-only", rev])
            revs[rev]["files_in_tree"] = len(tree.split())
        os.unlink(pat)
    return {"pushed_revision_before_rewrite": pushed, "by_revision": revs}


def extra_pattern_demo():
    """The hook's --extra hook-in, aimed at absolute development-host paths still present in tracked files."""
    with tempfile.NamedTemporaryFile("w", suffix=".pat", delete=False) as f:
        f.write("/home/[a-z0-9_-]+/\n")
        pat = f.name
    env = dict(os.environ, JEVLET_SCAN_EXTRA=pat)
    p = subprocess.run(["bench/scan_private.sh"], cwd=ROOT, capture_output=True, text=True, env=env)
    os.unlink(pat)
    locs = [l for l in p.stdout.splitlines() if l.endswith("[extra pattern]")]
    return {"pattern": "/home/<user>/ absolute paths", "exit": p.returncode, "hits": len(locs),
            "files": sorted({l.split(":")[0] for l in locs})}


def main():
    rec = {"date": dt.date.today().isoformat(), "produced_by": "python -m bench.hygiene_audit",
           "sanitizer": {"drop_keys": ["url", "base_url", "endpoint", "host", "server", "address"],
                         "public_device_name": "@sw1", "names_file": "data/raw/private.json (gitignored, untracked)",
                         "names_loaded": None},
           "dataset": dataset_counts(), "markers_in_published_data": markers(), "generative_runs": run_files(),
           "review_log": review_log(), "git": git_shape(), "pushes": pushes_from_transcript(),
           "private_census": private_census(), "incident": incident(),
           "hook_on_this_repo": hook_runs(), "hook_extra_pattern_demo": extra_pattern_demo(),
           "hook_on_synthetic_repo": synthetic()}
    m = re.search(r"names=(\d+)", rec["hook_on_this_repo"]["tree"]["summary"])
    rec["sanitizer"]["names_loaded"] = int(m.group(1)) if m else 0
    out = ROOT / "results" / f"hygiene-audit-{rec['date']}.json"
    json.dump(rec, open(out, "w"), indent=1)
    print(json.dumps(rec, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
