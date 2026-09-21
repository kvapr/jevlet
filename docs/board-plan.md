# Board measurements (Qualcomm IQ-9075 EVK) — plan, pending approval

Nothing in this plan changes the production serving state; all runs are on the board's CPU next to
the running `gx` container, and every artifact is copied off afterwards.

1. **Encoder latency on the Kryo cores** — `python -m bench.latency_cpu --threads 1,4` needs torch
   for the rlcd/laya/typed_decisions SDKs; the board's miniconda may lack torch. Fallback: a
   torch-free ONNX runner for openJev-verdict (GLiClass input format + `onnxruntime`), ~150 lines,
   which is also the deployable form.
2. **Readout latency** — start `llama-server` (build-cpu, aarch64) on a spare port with the 4B Q4_0
   GGUF and run `python -m bench.latency_readout --n 30`; expected: state-only prefill at 53-74 tok/s,
   i.e. 0.5-1.5 s per question, 30-60 ms/token decode irrelevant (one token).
3. **The whole battery on the board** — optional; accuracy does not change with the host, only the
   latency column does. Run the router battery once (`--splits paraphrase --limit 65`) to confirm
   identical answers to the Grace host (the Q4_0 GGUF is byte-identical).
4. **Co-tenancy** — measure an agent turn through `gx` while the CPU readout is running, to record
   the interference (expected small: the compiled path is NPU-bound, prefill on HTP).

Estimated board time: 1-2 hours. Requires: ssh to the board, ~3 GB free disk for the GGUF, no
service changes.
