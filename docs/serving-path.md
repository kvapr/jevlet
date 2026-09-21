# Serving-path facts that bound the design (measured 2026-09-21)

| path | on the board today | logprobs | prefix cache | `max_tokens` | typed decision possible |
|---|---|---|---|---|---|
| GenieX v0.4.0, compiled W4A16 bundle (production, NPU) | yes, `:18181` | **none** (`"logprobs": null` on a `logprobs:true, top_logprobs:5` request) | none (`cached_tokens: 0`; `GenieDialog_reset` clears KV per query) | ignored (3 tokens returned for `max_tokens: 1`) | text label only — no probability, no confidence, no calibration |
| GenieX 0.5.1-alpha (llama.cpp backend `-c npu`) | evaluation binary | no `n_probs` plumbing in the Go front end (binary strings: 0 hits) | unknown | honored | text label only |
| upstream `llama-server` + Hexagon backend | built (`pkg-snapdragon.tar.gz`), **not deployed**; single NPU tenant, cannot coexist with `gx` | yes (`n_probs`, `logprobs`, `logit_bias`, grammar); the lm-head is computed on the CPU on every path (`ggml-hexagon.cpp` refuses the lm-head matmul) | yes (`cache_prompt`, slot save/restore; HTP-resident KV reuse unverified) | honored | yes |
| `llama-server` on the board CPU | can run beside `gx` | yes | yes | honored | yes, at CPU prefill speed (~53-74 tok/s for the 4B) |
| small encoder (ONNX Runtime / torch) on the board CPU | can run beside `gx` | n/a (direct distribution) | n/a | n/a | yes, tens to hundreds of ms |

Consequences:

- A System One layer for the production deployment must run on the CPU (encoder, or a small
  decoder readout), or the agent itself must move to the upstream llama-server path (GGUF Q4_0 on
  HTP: 98.5% on the canonical bench but 14.5 s/turn vs 7.3 s compiled).
- With prefix caching, a readout question costs the state's tokens only: the question legend is
  the constant system prompt. On an uncontended host the per-request wall time is dominated by
  the ~20-80 state tokens plus one decoded token.
- On the compiled path, a text-label CHOICE from the same 4B (measured live on the 195 paraphrases)
  is *worse* than its own tool call: 88.7% vs 93.3% on the tool, 70.8% vs 83.1% on the command,
  with 7% of answers flipping when the option order is reversed. Routing through the generative
  model does not buy accuracy; the readout buys a probability.

Board hygiene note: `docker.service` on the EVK is disabled and socket-activated, so after a boot
the `gx` container is down until something touches Docker; `lance-local` then runs against a dead
backend. `systemctl enable docker` or a `Requires=docker.service` on the agent unit fixes it.
