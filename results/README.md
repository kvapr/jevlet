# Result receipts

Every number in the paper, blog and results page is derived from these files by
`python -m bench.analyze` (per-file summaries under `summary/`, figures under `figures/`) and
`python -m bench.print_summary`.

| file pattern | produced by | contents |
|---|---|---|
| `router-<backend>-<tag>.json` | `bench.run_readout`, `bench.run_encoders`, `train.eval_bundle` | `meta.rows` (the evaluated rows with labels), `meta.questions` (kind + options), `results[qid][]` = per-row `probs` (aligned with options), `logits` (pre-softmax scores), `label_mass`, `latency_s`, `per_rotation` (readout only) |
| `gate-<backend>-<tag>.json` | same runners with `--gate` | per pair: `input`, `command`, `params`, `kind` (real / gold / swap_op / wrong_port), `label`, `true_command`, `dangerous`, `p_yes`, `logits`; real pairs also carry `e2e` and `miss_kind` |
| `summary/<file>.json` | `bench.analyze` | accuracy, NLL, Brier, ECE (raw and out-of-fold temperature-scaled with folds grouped by seed request), selective accuracy at Jev's thresholds, per-style accuracy, confusions, order-flip rate, latency; gate: AUROC, real-miss detection by miss kind, dangerous-swap acceptance |
| `summary/*-heldout.json` | `bench.analyze --groups <split_manifest>` | the same, restricted to the held-out groups of a fine-tune split, for like-for-like comparison |
| `latency-readout.json` | `bench.latency_readout` | quiet-host readout latency per question: wall, evaluated prompt tokens, server prompt time |
| `latency-cpu*.json` | `bench.latency_cpu` | encoder latency at fixed thread counts (`-board` = on the QCS9075) |
| `figures/` | `bench.analyze`, `bench.figures` | reliability diagrams, risk–coverage curves, miss anatomy, comparison bars |

Tags: `qwen3-4b-q4_0-cpu` = readout of Qwen3-4B-Instruct-2507 Q4_0 on the Grace CPU (`-rot3` = three
option rotations averaged; `-authored` = the authored rows; `-all` = merged); `qwen3.5-0.8b-*` = the
tiny decoder (`-v2` = thinking disabled); `*-zeroshot` = open encoders as shipped (`-short` / `-bare`
= description mode); `*-all` = zero-shot on all 519 rows; `ft-*` = fine-tuned bundles on their held-out groups.
