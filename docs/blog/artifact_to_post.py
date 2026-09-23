#!/usr/bin/env python3
"""Turn a published artifact page into a standalone post for the personal site.

    python3 docs/blog/artifact_to_post.py [slug ...] [--site DIR]

Each artifact page in docs/artifact/ was written as a companion to the study's results page.
A standalone post needs three things changed: the site's own shell and stylesheet instead of the
artifact one, a self-contained opening instead of "companion to the results page", and public
links instead of the private claude.ai artifact URLs. Everything else, including the inline SVG
figures and the tables, is carried over unchanged.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "docs" / "artifact"
BLOG = "https://kvapr.github.io/posts/jevlet-system-one-on-the-edge/"
REPO = "https://github.com/kvapr/jevlet"

# the study's own pages, published privately as artifacts, have no public URL; point at the
# write-up that covers the same ground.
LINKS = {
    "https://claude.ai/artifact/BBMh93V3pxKYzSMe7UcpN3": BLOG,
    "https://claude.ai/artifact/3cL2PYgWi5ti1gRXScxhKa": BLOG,
}

# applied to every page: the footer's "companion to ..." opening, which carries links
COMMON = [
    (r'Companion to <a href="https://claude\.ai/artifact/[^"]*">the Jevlet results page</a>'
     r'(?: and <a href="[^"]*">the (?:encoder )?fine-tune page</a>)?'
     r'(?:; the fine-tune it refers to is on <a href="[^"]*">its own page</a>)?\.',
     f'Part of a study that put typed, calibrated decisions next to a 4B agent on a Qualcomm board '
     f'(<a href="{BLOG}">the write-up</a>).'),
]

PAGES = {
    "serving-path-logits": dict(
        slug="where-the-logits-go",
        date="September 2026",
        desc="A typed decision needs the probability distribution at the first generated position. On a Qualcomm board the production runtime returns none, the one NPU path that would cannot share the device, and what remains is a CPU server beside the running agent. Five serving paths, what each exposes, what the text-label fallback costs, and a porting checklist.",
        title="Where the logits go: what each way of running a 4B on a Qualcomm board exposes to a decision layer",
        subs=[
            (r'''<b>Read this with the main page\.</b>.*?goes into the encoder\.''',
             f'<b>Where this comes from.</b> This is one note out of a study that served typed, calibrated decisions locally beside an offline network-management assistant on a Qualcomm QCS9075 board: <a href="{BLOG}">the write-up</a> holds the accuracy and calibration comparison, and <a href="{REPO}">the repository</a> holds the code and the receipts.'),
        ],
    ),
    "tutorial-confirm-gate": dict(
        slug="calibrated-confirm-gate",
        date="September 2026",
        desc="Eight steps that turn a local instruction-tuned model into one calibrated yes/no question on your agent's confirm dialog: the llama-server flags, the question as a NOUL, reading the distribution, labelling your own pairs, grouped-fold calibration, the operating point, the hook, and five pitfalls with measured numbers.",
        title=None,
        subs=[
            (r'''<b>Read this with the <a href="[^"]*">results page</a>\.</b> That page compares backends; this one is''',
             f'<b>Where the numbers come from.</b> They are measurements from a study that served typed, calibrated decisions beside an offline network-management assistant on a Qualcomm QCS9075 board (<a href="{BLOG}">the write-up</a>, <a href="{REPO}">the code</a>). That study compared four backends; this page is'),
        ],
    ),
    "order-sensitivity": dict(
        slug="option-order-flips",
        date="September 2026",
        desc="How often a 4B and a 0.8B readout change their answer when only the option order changes, and why they flip for opposite reasons: the larger model on content and confidently, the smaller one because it likes the letter B. What averaging rotations buys, what it costs, and why temperature cannot fix it.",
        title=None,
        subs=[
            (r'''Read it with the <a href="[^"]*">results page</a>, which holds the accuracy and calibration numbers the rotations feed into\.''',
             f'The readouts and the rows come from a study that served typed, calibrated decisions beside an offline network-management assistant on a Qualcomm board: <a href="{BLOG}">the write-up</a> holds the accuracy and calibration numbers the rotations feed into, and <a href="{REPO}">the repository</a> holds the receipts.'),
        ],
    ),
    "gate-wording": dict(
        slug="gate-question-wording",
        date="September 2026",
        desc="The same yes/no gate question in two phrasings, against the same model and the same 400 pairs. The ranking does not move and the operating point does, a temperature provably cannot follow it, and most of the shift a first pass had blamed on the sentence turns out to belong to the answer legend.",
        title="A threshold belongs to a sentence, not to a model",
        subs=[
            (r'''<b>What ran where\.</b>''',
             f'<b>Where this comes from.</b> The gate guards device writes in a study that served typed, calibrated decisions beside an offline network-management assistant on a Qualcomm board (<a href="{BLOG}">the write-up</a>, <a href="{REPO}">the code</a>).'),
        ],
    ),
}


def site_style(site: Path) -> str:
    """The site's own post stylesheet, taken from an existing post so the two cannot drift."""
    ref = site / "posts" / "jevlet-system-one-on-the-edge" / "index.html"
    m = re.search(r"<style>.*?</style>", ref.read_text(), re.S)
    if not m:
        raise SystemExit(f"no <style> block in {ref}")
    return m.group(0)


COMPAT = """<style>
  /* components carried over from the study's results page, on the site's own tokens */
  :root {
    --ground: var(--bg); --ink: var(--fg); --rule: var(--border);
    --accent-soft: #e2f1e8; --ok: #2e7d5b; --ok-soft: #e2f1e8;
    --warn: #c7831c; --warn-soft: #f7ecd6; --bad: #b5432f; --bad-soft: #f6e2dd;
    --readout: #3253dc; --encoder: #c98a2b; --finetuned: #a8404b; --generative: #6b7378;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --accent-soft: #173126; --ok: #63c08f; --ok-soft: #173126;
      --warn: #e2a64c; --warn-soft: #3a2c15; --bad: #e07a68; --bad-soft: #3d201c;
      --readout: #8fa6ff; --encoder: #e2a64c; --finetuned: #e28a93; --generative: #9aa6ab;
    }
  }
  .eyebrow { display: none; }
  th { text-align: left; font: 500 .76rem/1.3 var(--mono); letter-spacing: .05em; text-transform: uppercase;
       color: var(--muted); padding: .45rem .75rem .45rem 0; border-bottom: 1px solid var(--fg); white-space: nowrap; }
  td.n, th.n { text-align: right; font-family: var(--mono); font-size: .92em; }
  tr.lead td { font-weight: 600; }
  .kv { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: .6rem; margin: 1.2rem 0; }
  .kv div { background: var(--surface-2); border: 1px solid var(--border); border-radius: 8px; padding: .7rem .8rem; }
  .kv b { display: block; font: 600 1.25rem/1.15 var(--mono); }
  .kv span { font-size: .82rem; color: var(--muted); }
  .note { border-left: 3px solid var(--accent); background: var(--surface-2); padding: .8rem 1rem;
          border-radius: 0 8px 8px 0; font-size: .95rem; margin: 1.4rem 0; }
  .card-demo { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 1rem 1.1rem; margin: 1.3rem 0; }
  .card-demo .req { font-weight: 500; }
  .card-demo .cmd { font-family: var(--mono); font-size: .82rem; background: var(--code-bg); padding: .6rem .8rem;
                    border-radius: 6px; margin: .6rem 0; white-space: pre-wrap; overflow-x: auto; }
  .check { display: flex; gap: .6rem; align-items: flex-start; font-size: .93rem; padding: .6rem .75rem;
           border-radius: 6px; margin-top: .6rem; color: var(--fg); }
  .check.bad { background: var(--bad-soft); border-left: 3px solid var(--bad); }
  .check.ok { background: var(--ok-soft); border-left: 3px solid var(--ok); }
  .check.warn { background: var(--warn-soft); border-left: 3px solid var(--warn); }
  .check b { font-family: var(--mono); font-weight: 500; white-space: nowrap; }
  .actions { display: flex; gap: .5rem; margin-top: .7rem; }
  .btn { font-size: .85rem; padding: .35rem .7rem; border-radius: 6px; border: 1px solid var(--border); background: var(--surface-2); }
  .btn.primary { background: var(--qc); color: #fff; border-color: var(--qc); }
  .pill { font: 500 .72rem var(--mono); padding: .1rem .45rem; border-radius: 999px; border: 1px solid var(--border);
          color: var(--muted); white-space: nowrap; }
  .swatch { display: inline-block; width: .7em; height: .7em; border-radius: 2px; margin-right: .4em; vertical-align: -1px; }
  .pending { color: var(--warn); font-family: var(--mono); font-size: .82rem; }
  figure { background: var(--surface); }
  figure svg { width: 100%; height: auto; display: block; }
  figure img { max-width: 100%; display: block; margin: 0 auto; background: #fff; border-radius: 6px; }
  main > p:first-of-type { margin-top: 0; }
</style>"""


def convert(name: str, cfg: dict, site: Path, style: str) -> tuple[Path, str, str]:
    src = (ART / f"{name}.html").read_text()
    head, _, rest = src.partition("<main>")
    body, _, _ = rest.rpartition("</main>")

    # page-specific styles only: the first <style> in the head is the shared artifact shell
    extra = re.findall(r"<style>.*?</style>", head, re.S)[1:]

    for pat, new in cfg["subs"] + COMMON:
        body, n = re.subn(pat, lambda _m, _n=new: _n, body, flags=re.S)
        if n != 1:
            raise SystemExit(f"{name}: pattern {pat[:60]!r} matched {n} times, expected 1")
    for old, new in LINKS.items():
        body = body.replace(old, new)
    if "claude.ai/artifact" in body:
        raise SystemExit(f"{name}: a private artifact link survived")

    body = re.sub(r'<div class="eyebrow">.*?</div>\s*', "", body, count=1, flags=re.S)
    title = cfg["title"]
    if title:
        body = re.sub(r"<h1>.*?</h1>", f"<h1>{title}</h1>", body, count=1, flags=re.S)
    else:
        title = re.search(r"<h1>(.*?)</h1>", body, re.S).group(1)
    body = body.replace('<p class="lede">', '<p class="subtitle">', 1)

    out_dir = site / "posts" / cfg["slug"]
    out_dir.mkdir(parents=True, exist_ok=True)
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{re.sub(r"<[^>]+>", "", title)}</title>
{style}
{COMPAT}
{chr(10).join(extra)}
</head>
<body>
<main>
<p style="margin:0 0 1.5rem"><a href="../../" style="color:var(--muted);text-decoration:none;font-size:.9rem">&#8592; home</a></p>
{body.strip()}
</main>
</body>
</html>
"""
    (out_dir / "index.html").write_text(page)
    for png in sorted(set(re.findall(r'src="([^"/]+\.png)"', body))):
        shutil.copyfile(ART / png, out_dir / png)
    return out_dir, re.sub(r"<[^>]+>", "", title), cfg["desc"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("slugs", nargs="*", default=sorted(PAGES), help="artifact page names (default: all configured)")
    ap.add_argument("--site", default=str(Path.home() / "projects" / "kvapr.github.io"))
    a = ap.parse_args()
    site = Path(a.site)
    style = site_style(site)
    for name in (a.slugs or sorted(PAGES)):
        if name not in PAGES:
            raise SystemExit(f"unknown page {name}; configured: {', '.join(sorted(PAGES))}")
        out, title, desc = convert(name, PAGES[name], site, style)
        print(f"{name:24s} -> {out.relative_to(site)}  ({(out / 'index.html').stat().st_size:,} bytes)")
        print(f"   title: {title}")
        print(f"   desc : {desc[:110]}...")


if __name__ == "__main__":
    main()
