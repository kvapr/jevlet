# jevlet — design notes

## The problem, concretely

An offline network-management assistant runs a 4-billion-parameter instruction-tuned model on
an embedded NPU (Qwen3-4B-Instruct-2507, W4A16, 4K context, compiled for a Qualcomm QCS9075).
It turns an operator's sentence into one tool call; a human confirms every write before it
reaches the switch. Measured (65 canonical requests, 195 paraphrases): 92-97% end-to-end on
canonical phrasing, 77.4% on paraphrases, 61.5% on indirect ("symptom-first") phrasing.

Two properties of that generative path are the actual product risk, not the headline accuracy:

1. **Its errors are structured and some are dangerous.** Of 44 paraphrase misses, 20 are
   confusions between semantically adjacent commands — 12 access-VLAN vs trunk, 4 VLAN
   config vs membership, 2 PoE-off vs port-shutdown, 1 speed vs shutdown, 1 PoE mode vs
   limit — and 13 are "diagnose instead of read" (defensible), 10 are format slips (a bare
   command name in the tool slot; deterministically recoverable).
2. **It emits no probability.** A tool call is a one-hot event. The confirm card cannot tell a
   confident call from a coin flip, and the compiled NPU serving path exposes no logits at all
   (verified: GenieX v0.4.0 returns `logprobs: null`, ignores `max_tokens`, has no prefix cache),
   so "read the logits" is not available in production.

## The idea (borrowed from Jev, kept as a concept)

TypeSafe's Jev reframes such decisions as typed questions over a state — NOUL (yes/no),
CHOICE (one of K), SCORE (ordered levels) — answered in one parallel pass with calibrated
probabilities, no text generation. The interface is the useful part: an application thresholds
a distribution instead of parsing a sentence, and a confidence axis exists at all.

Jev itself is cloud-only, so on an air-gapped board the concept has to be served locally.
Three local realizations are compared here, all behind the same `jevlet` interface:

| backend | what it is | probability semantics |
|---|---|---|
| `llamacpp` readout | the same 4B, one prefill + the next-token distribution over single-token labels (A..O / 0..9 / Yes/No) via `llama-server` `logprobs` | over-confident and order-biased until temperature-scaled and permutation-averaged |
| open encoder decision models | openJev-verdict-2.0 (151M, ModernBERT-base, GLiClass), Laya (421M, RLCD), open-jev-deberta-v3-large (0.4B) — zero-shot | trained for calibration on *their* domains; out of domain here |
| fine-tuned small encoder | ModernBERT-base / DeBERTa-v3-base trained on the netops rows (grouped split), CE + Brier | calibrated on this domain; measured out-of-fold |
| rules | regex / lookup | one-hot, no epistemics; the floor and the timeout fallback |

## Where it sits in the agent turn

```
operator text ──► [router battery: tool / command / is_write / touches_poe] ──► tool profile,
                                                                               clarify hint
                 ──► generative 4B proposes a call ──► [gate battery: matches_request] ──► confirm card
                                                                                           annotation
                                                                            (human clicks; nothing auto-executes)
```

The decision layer never executes anything. Its outputs are (a) which tool schemas to load
into the 4K context, (b) a one-line "check" on the confirm card with a probability, (c) a
hint to state the reading explicitly when the request is ambiguous. Jev's own guidance
(">0.9 act automatically") is deliberately NOT applied to device writes.

## Question batteries

- **router** (state = the request): `tool` CHOICE over 8 (incl. none), `command` CHOICE over the
  23-command catalog + none, `is_write` NOUL, `touches_poe_power` NOUL. Confusable pairs are
  named in the instructions.
- **gate** (state = request + proposed call + one-line description): `matches_request` NOUL.
  Labels come from the generative model's *real* calls (its e2e correctness) plus synthetic
  negatives (swapped operation, wrong port).

## What is measured

Accuracy per style (canonical / casual / verbose / indirect / adversarial), ECE (raw and
out-of-fold temperature-scaled, folds grouped by seed request), Brier, NLL, selective accuracy
at the Jev thresholds (0.5 / 0.7 / 0.9), confusable-pair confusion counts, order-flip rate
under option rotation, real-miss detection rate of the gate, dangerous-swap acceptance, and
CPU latency per call at 1 and 4 threads on an aarch64 host (board numbers where available).

## Non-goals

- Replacing the generative tool caller. The readout experiment shows why: a text-label CHOICE
  from the same model picks tools *less* accurately than its own tool call (88.7% vs 93.3%).
- Auto-execution on confidence. The human confirm is the product's invariant.
- Bundling a model with the application. The layer is optional; the rules backend is the floor.
