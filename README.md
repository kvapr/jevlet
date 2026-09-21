# jevlet

**A small, calibrated System One decision layer for offline edge agents.**

`jevlet` takes the idea behind TypeSafe's Jev — ask typed questions (NOUL yes/no, CHOICE one-of-K,
SCORE ordered levels) about a piece of state and get back probability distributions instead of
generated text — and serves it *locally*, on the CPU next to a small generative agent, for one
concrete workload: an offline network-management assistant that drives a managed PoE switch from
a 4B model on an embedded NPU.

It is a study first and a library second. The library exists so that the same application code
can run against any of four backends that return the *same epistemic object* (a distribution the
application thresholds), while the study measures how much that distribution can be trusted
from each of them:

| backend | what runs | probability comes from |
|---|---|---|
| `llamacpp` readout | the resident instruction-tuned LM (Qwen3-4B), one prefill + next-token logprobs over single-token labels via `llama-server` | the LM's own softmax; needs temperature scaling and option-rotation averaging |
| open encoder decision models | [openJev-verdict-2.0](https://huggingface.co/heman10x/rlcd-modernbert-151m) (151M), [Laya](https://huggingface.co/convaiinnovations/laya) (421M), [open-jev-deberta-v3-large](https://huggingface.co/com-kotobalabs/open-jev-deberta-v3-large) (0.4B) | their training; zero-shot here |
| fine-tuned encoder | ModernBERT-base / DeBERTa-v3-base trained on the netops rows with the [typed-decisions](https://github.com/kotoba-lang/typed-decisions) recipe | CE + Brier training, out-of-fold temperature |
| `rules` | regexes and lookup tables | none (one-hot); the floor and the timeout fallback |
| `typesafe` | the cloud Jev API (reference; not usable offline) | RLCD, vendor-calibrated |

The wire shape follows the public Jev API (`state` + `questions` → `answers` with `probabilities`
and `confidence`), so a `jevlet` backend can sit behind code written for Jev.

## Why

The generative tool caller on the board is accurate on canonical phrasing (92-97% end-to-end)
and weaker on paraphrases (77.4%; 61.5% on indirect phrasing). Its misses are not random: 20 of
44 are confusions between adjacent commands, several of them dangerous (cut PoE power vs shut the
interface; put a port in one VLAN vs make it a trunk). And it emits no probability — a tool call
is a one-hot event — so the human confirm gate cannot tell a confident call from a coin flip. On
the compiled NPU path there are no logits to read either (GenieX returns `logprobs: null`).

A decision layer that answers *typed* questions about the request and the proposed call, with a
probability whose calibration has been measured on this domain, gives the confirm gate the one
thing it lacks: a confidence axis. It never executes anything.

See [docs/DESIGN.md](docs/DESIGN.md) for where the layer sits in the agent turn and what is
measured, and [results/summary](results/summary) for the numbers.

## Install

```sh
pip install -e .                 # core: numpy, pyyaml, requests
pip install -e ".[encoders,bench]"   # torch, transformers, onnxruntime, scipy, matplotlib
```

## Use

```python
from jevlet.backends.llamacpp import LlamaCppReadout
from jevlet.schema import Choice, Noul

be = LlamaCppReadout("http://127.0.0.1:18999", rotations=3,
                     domain_hint="Operators mention the switch as @sw1; 'port N' means interface 1/N.")
res = be.decide("@sw1 kill the power to the camera on port 3", {
    "op": Choice("Which operation does the operator want?", {
        "poe_power_off": "turn PoE power off on the port; the link stays up",
        "port_admin_down": "administratively shut the interface; link and power go down",
        "other": None}),
    "is_write": Noul("Does the request ask to change a setting?"),
})
print(res.to_wire(normalize_questions(...)))   # {"answers": {"op": {"choice": ..., "probabilities": {...}, "confidence": ...}, "is_write": {"noul": 0.97}}}
```

Every `Answer` also carries `logits` (log-probabilities) so you can re-calibrate:

```python
from jevlet.calibrate import cv_temperature   # grouped out-of-fold temperature scaling
from jevlet.metrics import summarize          # accuracy, NLL, Brier, ECE, selective accuracy
```

## The dataset: `data/netops`

Operator requests to a managed PoE switch, sanitized (the device is `@sw1`; no addresses):

- `canonical.yaml` — 65 seed requests in four difficulty levels (status / on-off / one-parameter / multi-parameter)
- `paraphrase.yaml` — 195 paraphrases of the seeds in three styles (A casual, B verbose, C indirect / symptom-first)
- `authored/` — rows written for this study and verified by an independent pass: adversarial confusables
  (PoE-off vs shutdown, access vs trunk, ...), diagnose / documentation / out-of-scope / other-tool intents,
  each with `is_write`, `touches_poe_power`, `ambiguity` (0-3), `alt_ok`, `blast` labels;
  `train_paraphrases.yaml` are label-preserving paraphrases used only for training
- `catalog.json` — the 33 flat tool/command names with descriptions
- `generative_runs/` — the generative tool-calling runs of several models on the same requests (one-hot calls, latencies)

Evaluation splits are always grouped by seed request; paraphrases of a seed never cross a split.
The review log of every authored row is in [docs/data-review-log.md](docs/data-review-log.md).

## Reproduce

```sh
python -m bench.netops build                                   # sanitized dataset from data/raw (private)
llama-server -m Qwen3-4B-Instruct-2507-Q4_0.gguf --port 18999 --jinja
python -m bench.run_readout --url http://127.0.0.1:18999 --tag qwen3-4b --rotations 3       # E1 router battery
python -m bench.run_readout --url http://127.0.0.1:18999 --tag qwen3-4b --gate --run data/netops/generative_runs/g3-2507-npu-para195.json
python -m bench.run_encoders --backend verdict --models <models> --repo <Verdict-open-jev> --tag verdict   # E2
bash train/run_finetune.sh 4 && python -m train.eval_bundle --bundle runs/modernbert-base-desc/bundle ...    # E3
python -m bench.analyze results/router-*.json results/gate-*.json                                          # tables + figures
python -m bench.latency_cpu --models <models> --repo <Verdict-open-jev> --threads 1,4                       # CPU latency
```

## Headline numbers (September 2026)

| | 4B readout | fine-tuned 150M encoder (round 2) | best zero-shot encoder |
|---|---:|---:|---:|
| router: tool / command, 126 held-out rows | 84.9% / 69.8% | **88.1% / 77.8%** | 61.7% / 51.4% (Laya-td, all rows) |
| gate: AUROC, 939 pairs | **0.952** | 0.53–0.65 (all encoder variants) | 0.757 (Laya) |
| gate: real dangerous confusions flagged at p<0.5 | 15 / 20 | — | 1 / 20 |
| board CPU, one NOUL / one gate check | 1.24 s / 3.1 s | 126 ms / 183 ms | 352 ms / 527 ms |

The division of labor this implies: an encoder for the per-turn router, a decoder readout for the
per-write gate. Full tables, calibration and figures: [results/summary](results/summary),
[docs/artifact](docs/artifact), the blog draft in [docs/blog](docs/blog), and the preprint in [paper](paper)
(`bash paper/build.sh` rebuilds `paper/main.pdf` and an arXiv source bundle).

## Status

Research code, September 2026. Numbers in `results/` are from an aarch64 Grace host (CPU), a GB10
GPU, and the Qualcomm QCS9075 board (files tagged `board`). Licensed Apache-2.0. The open models
above keep their own licenses (Apache-2.0); Qwen weights keep theirs.
