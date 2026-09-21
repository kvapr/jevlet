#!/usr/bin/env bash
# Render the blog markdown into the personal site's self-contained post format
# (kvapr.github.io/posts/<slug>/index.html): pandoc for the body, the site's post CSS shell around it.
#   bash docs/blog/build.sh [outdir]
set -euo pipefail
cd "$(dirname "$0")"
SLUG=jevlet-system-one-on-the-edge
OUT=${1:-site/$SLUG}
mkdir -p "$OUT"
TITLE=$(grep -m1 '^# ' "$SLUG.md" | sed 's/^# //')
SUBTITLE=$(grep -m1 '^\*' "$SLUG.md" | sed 's/^\*//; s/\*$//')
# body = everything after the title line and the italic subtitle line
BODY=$(sed '1{/^# /d}; 2,3{/^\*.*\*$/d}' "$SLUG.md" | pandoc -f gfm -t html5 --no-highlight --wrap=none)
cat > "$OUT/index.html" <<HTML
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>$TITLE</title>
<style>
  :root {
    --bg: #ffffff; --fg: #1a1d23; --muted: #5a6270; --accent: #0b6e4f;
    --qc: #3253dc; --border: #d8dde4; --loss: #b5382e;
    --code-bg: #f3f5f8; --surface: #ffffff; --surface-2: #f3f5f8;
    --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0f1216; --fg: #e6e9ee; --muted: #9aa4b2; --accent: #4dd2a5;
      --qc: #8fa6ff; --border: #2a323c; --loss: #ea7d71;
      --code-bg: #161b22; --surface: #141920; --surface-2: #1a2028;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--fg); font: 17px/1.65 -apple-system, "Segoe UI", Roboto, "Helvetica Neue", sans-serif; }
  main { max-width: 720px; margin: 0 auto; padding: 2.5rem 1.25rem 4rem; }
  h1 { font-size: 1.7rem; line-height: 1.25; margin: 0 0 .25rem; }
  h2 { font-size: 1.25rem; margin: 2.2rem 0 .6rem; padding-top: .9rem; border-top: 1px solid var(--border); }
  h3 { font-size: 1.05rem; margin: 1.8rem 0 .4rem; }
  .subtitle { color: var(--muted); margin: 0 0 1.25rem; }
  p { margin: .8rem 0; }
  a { color: var(--qc); }
  ul, ol { padding-left: 1.4rem; }
  li { margin: .35rem 0; }
  blockquote { border-left: 3px solid var(--accent); margin: 1rem 0; padding: .2rem 1rem; color: var(--muted); }
  code { font-family: var(--mono); font-size: .88em; font-variant-ligatures: none; font-feature-settings: "liga" 0, "calt" 0; }
  p code, li code, td code { background: var(--code-bg); padding: .05em .3em; border-radius: 3px; }
  pre { background: var(--code-bg); border: 1px solid var(--border); border-radius: 8px; padding: .8rem 1rem; overflow-x: auto; font-family: var(--mono); font-size: .82rem; line-height: 1.5; font-variant-ligatures: none; font-feature-settings: "liga" 0, "calt" 0; }
  pre code { font-size: 1em; background: none; padding: 0; }
  .tablewrap { overflow-x: auto; margin: 1rem 0 1.25rem; }
  table { border-collapse: collapse; width: 100%; font-size: .93rem; line-height: 1.45; font-variant-numeric: tabular-nums; }
  thead th { text-align: left; font-size: .76rem; letter-spacing: .05em; text-transform: uppercase; color: var(--muted); padding: .45rem .75rem .45rem 0; border-bottom: 1px solid var(--fg); white-space: nowrap; }
  tbody td { vertical-align: top; padding: .5rem .75rem .5rem 0; border-bottom: 1px solid var(--border); }
  tbody tr:last-child td { border-bottom: 1px solid var(--fg); }
  th[style*="right"], td[style*="right"] { text-align: right; }
  th:last-child, td:last-child { padding-right: 0; }
  figure { margin: 1.5rem 0; padding: .75rem .75rem .5rem; background: var(--surface); border: 1px solid var(--border); border-radius: 10px; }
  figure img { max-width: 100%; display: block; margin: 0 auto; background: #ffffff; border-radius: 6px; }
  figcaption { font-size: .88rem; color: var(--muted); margin-top: .6rem; line-height: 1.5; }
  figcaption .figno { font-weight: 600; color: var(--fg); }
  figure svg { width: 100%; height: auto; display: block; color: var(--fg); }
  svg text { font-size: 12.5px; fill: currentColor; font-family: inherit; }
  svg text.h { font-weight: 600; font-size: 13px; }
  svg text.b { font-weight: 600; }
  svg text.s { fill: var(--muted); font-size: 11.5px; }
  svg text.mono { font-family: var(--mono); font-size: 11px; }
  svg text.tag { fill: var(--accent); font-family: var(--mono); font-size: 9.5px; letter-spacing: .08em; }
  svg text.lbl { fill: var(--muted); font-size: 11px; }
  svg .lane { fill: var(--surface-2); stroke: var(--border); stroke-width: 1; }
  svg .fp, svg .dec { fill: var(--surface); stroke: var(--accent); stroke-width: 1.5; }
  svg .box { fill: var(--surface-2); stroke: var(--border); stroke-width: 1; }
  svg .end { fill: var(--surface-2); stroke: var(--accent); stroke-width: 1.5; }
  svg .flow { fill: none; stroke: currentColor; stroke-width: 1.5; }
  svg .bar { fill: var(--qc); }
  svg .warn { fill: var(--loss); }
  hr { border: 0; border-top: 1px solid var(--border); margin: 2rem 0; }
</style>
</head>
<body>
<main>
<p style="margin:0 0 1.5rem"><a href="../../" style="color:var(--muted);text-decoration:none;font-size:.9rem">&#8592; home</a></p>

<h1>$TITLE</h1>
<p class="subtitle">$SUBTITLE</p>
$BODY
</main>
</body>
</html>
HTML
# images alone in a paragraph become captioned figures (the alt text is the caption)
python3 - "$OUT/index.html" <<'PY'
import re, sys, html, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
def fig(m):
    src, alt = m.group(1), html.unescape(m.group(2))
    cap = html.escape(alt)
    cap = re.sub(r'^(Figure \d+\.)\s*', r'<span class="figno">\1</span> ', cap)
    return f'<figure><img src="{src}" alt="{html.escape(alt, quote=True)}"><figcaption>{cap}</figcaption></figure>'
s = re.sub(r'<p><img src="([^"]+)" alt="([^"]*)" /></p>', fig, s)
p.write_text(s)
PY
# tables scroll sideways on phones instead of widening the page
sed -i 's|<table>|<div class="tablewrap"><table>|g; s|</table>|</table></div>|g' "$OUT/index.html"
cp -f ../../results/figures/reliability-llamacpp-readout-command.png "$OUT/reliability-readout-command.png"
cp -f ../../results/figures/risk-coverage-llamacpp-readout-gate.png "$OUT/risk-coverage-readout-gate.png"
rm -f "$OUT/miss-anatomy.png"
echo "wrote $OUT/index.html ($(wc -c < "$OUT/index.html") bytes)"
