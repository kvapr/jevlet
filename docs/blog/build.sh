#!/usr/bin/env bash
# Render the blog markdown into the personal site's self-contained post format
# (kvapr.github.io/posts/<slug>/index.html): pandoc for the body, the site's CSS shell around it.
set -euo pipefail
cd "$(dirname "$0")"
SLUG=jevlet-system-one-on-the-edge
OUT=${1:-site/$SLUG}
mkdir -p "$OUT"
BODY=$(pandoc "$SLUG.md" -f gfm -t html5 --no-highlight --wrap=none | sed '1,/<\/h1>/d')
TITLE=$(grep -m1 '^# ' "$SLUG.md" | sed 's/^# //')
cat > "$OUT/index.html" <<EOF
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>$TITLE</title>
<style>
  :root { --bg: #ffffff; --fg: #1a1d23; --muted: #5a6270; --accent: #0b6e4f; --qc: #3253dc; --border: #d8dde4; }
  @media (prefers-color-scheme: dark) { :root { --bg: #0f1216; --fg: #e6e9ee; --muted: #9aa4b2; --accent: #4dd2a5; --qc: #8fa6ff; --border: #2a323c; } }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--fg); font: 17px/1.65 -apple-system, "Segoe UI", Roboto, "Helvetica Neue", sans-serif; }
  main { max-width: 720px; margin: 0 auto; padding: 2.5rem 1.25rem 4rem; }
  h1 { font-size: 1.7rem; line-height: 1.25; margin: 0 0 .25rem; }
  h2 { font-size: 1.25rem; margin: 2.2rem 0 .6rem; padding-top: .9rem; border-top: 1px solid var(--border); }
  .subtitle { color: var(--muted); margin-bottom: 2rem; }
  p { margin: .8rem 0; } a { color: var(--qc); } ul { padding-left: 1.4rem; } li { margin: .35rem 0; }
  blockquote { border-left: 3px solid var(--accent); margin: 1rem 0; padding: .2rem 1rem; color: var(--muted); }
  code { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: .88em; }
  pre { background: rgba(127,127,127,.08); padding: .8rem 1rem; border-radius: 6px; overflow-x: auto; font-size: .82rem; line-height: 1.45; }
  .tbl { overflow-x: auto; } table { border-collapse: collapse; font-size: .9rem; font-variant-numeric: tabular-nums; width: 100%; }
  th, td { text-align: left; padding: .4rem .7rem .4rem 0; border-bottom: 1px solid var(--border); vertical-align: top; }
  .fig { background: #ffffff; border: 1px solid var(--border); border-radius: 10px; padding: .5rem; margin: 1.5rem 0; overflow-x: auto; }
  .fig img { max-width: 100%; display: block; margin: 0 auto; }
</style>
</head>
<body>
<main>
<p style="margin:0 0 1.5rem"><a href="../../" style="color:var(--muted);text-decoration:none;font-size:.9rem">&#8592; home</a></p>
<h1>$TITLE</h1>
$BODY
</main>
</body>
</html>
EOF
# wrap tables for horizontal scroll on phones
sed -i 's|<table>|<div class="tbl"><table>|g; s|</table>|</table></div>|g' "$OUT/index.html"
cp -f ../../results/figures/reliability-llamacpp-readout-command.png "$OUT/reliability-readout-command.png" 2>/dev/null || true
cp -f ../../results/figures/miss-anatomy.png "$OUT/miss-anatomy.png" 2>/dev/null || true
echo "wrote $OUT/index.html"
