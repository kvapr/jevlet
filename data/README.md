# netops decision dataset

Operator requests to a managed PoE switch, with the tool call each request should produce and
typed labels for a System One decision layer. Built from the evaluation set of an offline
network-management assistant (a 4B model on a Qualcomm NPU) and extended for this study.

## Files (`data/netops/`)

| file | rows | what |
|---|---:|---|
| `canonical.yaml` | 65 | seed requests in four difficulty levels: status query, enable/disable, one-parameter setting, multi-parameter setting |
| `paraphrase.yaml` | 195 | every seed rephrased three ways: A casual, B verbose, C indirect / symptom-first; labels preserved |
| `authored/adversarial_confusables.yaml` | ~90 | style X: phrasings that sit between adjacent commands (PoE power off vs port shutdown, access VLAN vs trunk, speed vs shutdown, PoE mode vs limit, VLAN config vs membership); some deliberately ambiguous (`ambiguity` 2 with `alt_ok`) |
| `authored/intents_diagnose.yaml` | ~55 | symptom-first troubleshooting requests → `diagnose_port_issues`, plus borderline status reads and symptom-then-change rows |
| `authored/intents_docs_and_oos.yaml` | ~80 | documentation / how-to questions, out-of-scope messages (`none`), fleet-wide requests the single-device profile cannot serve |
| `authored/intents_other_tools.yaml` | ~50 | requests for the tools outside the 4-tool production profile: switch events, camera detections, cable-baseline calibration |
| `authored/train_paraphrases.yaml` | ~1000 | label-preserving paraphrases (A/B/C/T terse) of the seeds, used ONLY for training; drift-checked |
| `catalog.json` | 33 | flat tool/command names with one-line descriptions |
| `generative_runs/*.json` | 46 runs | the generative tool-calling runs (several models, serving paths, prompts) on the same requests: per-row call, correctness tiers, latency |

## Row schema

```yaml
- input: "@sw1 kill the power to the camera on port 3"
  expected_tool: configure_device_setting     # or query_device_info | diagnose_port_issues | search_documentation |
                                               # get_network_events | get_camera_detections | calibrate_cable_baseline | none
  expected_command: configure_poe_mode        # command_name under the two wrapper tools; "" otherwise
  expected_params: {port: "1/3", mode: disable}
  style: X                                    # canon | A | B | C | T | X
  base_input: "..."                           # the seed request this row derives from (grouping key for splits)
  is_write: true                              # changes a setting?
  touches_poe_power: true                     # about PoE power delivery rather than link/traffic/VLAN?
  dangerous_pair: poe_vs_shutdown             # which confusable pair the row stresses, or none
  ambiguity: 1                                # 0 one clear call · 1 minor gap · 2 two plausible readings · 3 unclear
  alt_ok: []                                  # other commands a careful operator would also accept
  blast: 3                                    # 0 none · 1 negligible · 2 brief self-recovering · 3 down until reverted · 4 segment outage
  notes: "..."
```

Canonical and paraphrase rows carry only the first six fields; `is_write` / `touches_poe_power`
are derived from the command for them, `ambiguity` is 0.

## Conventions

- The device is always `@sw1`; ports are written `1/N` in params; speeds `10|100|1000|auto`;
  duplex `full|half`; PoE modes `enable|disable`; watts integers; VLAN ids integers.
- `disable_port` shuts the interface (link AND power); `configure_poe_mode disable` cuts PoE
  only; `configure_access_vlan` = one untagged VLAN on an access port; `configure_trunk_allowed_vlans`
  = trunk with an allowed list. These are the pairs the labels are careful about.
- Splits are grouped by `base_input`. A paraphrase never appears in a different split than its seed.

## Provenance and review

Canonical and paraphrase rows were written by the application team in 2026-07/08. The authored
rows and training paraphrases were generated in 2026-09 by an LLM author and checked row-by-row
by an independent LLM verifier for label correctness, format, realism and duplicates; every
change is listed in [docs/data-review-log.md](../docs/data-review-log.md). Human review of the
authored rows is pending and is flagged as such in the study. No real device names, addresses or
credentials appear anywhere; the generative run files had their endpoint URLs removed.

License: Apache-2.0 (data and annotations).
