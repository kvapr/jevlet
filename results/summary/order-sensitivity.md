# Order sensitivity of the readout (bench.order_sensitivity)

Generated 2026-09-22. Definitions are in results/order-sensitivity.json.

| model | rows | question | k | R | flip rate | pairwise mean | single rotation r0 / r1 / r2 | averaged | majority | any rotation right | confident flips | first-shown wins vs gold share (chance) |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| Qwen3-4B | 260 base rows | tool | 8 | - | n/a | n/a | n/a | 96.5% | n/a | n/a | n/a | per_rotation not recorded in this receipt; averaged probabilities only |
| Qwen3-4B | 260 base rows | command | 24 | - | n/a | n/a | n/a | 84.6% | n/a | n/a | n/a | per_rotation not recorded in this receipt; averaged probabilities only |
| Qwen3-4B | 260 base rows | is_write | 2 | - | n/a | n/a | n/a | 98.1% | n/a | n/a | n/a | per_rotation not recorded in this receipt; averaged probabilities only |
| Qwen3-4B | 260 base rows | touches_poe_power | 2 | - | n/a | n/a | n/a | 98.8% | n/a | n/a | n/a | per_rotation not recorded in this receipt; averaged probabilities only |
| Qwen3-4B | 259 authored rows | tool | 8 | 3 | 36.3% | 25.1% | 64.5 / 64.5 / 63.3 | 68.3% | 68.3% | 74.9% | 93 / 94 | 16.7% vs 18.4% (12.5%) |
| Qwen3-4B | 259 authored rows | command | 24 | 3 | 29.3% | 22.4% | 60.2 / 57.1 / 60.2 | 60.2% | 59.8% | 68.3% | 75 / 76 | 5.7% vs 1.3% (4.2%) |
| Qwen3-4B | 259 authored rows | is_write | 2 | 1 | n/a | n/a | n/a | 85.7% | n/a | n/a | n/a | one rotation recorded (NOUL questions are not rotated); flips not measurable |
| Qwen3-4B | 259 authored rows | touches_poe_power | 2 | 1 | n/a | n/a | n/a | 95.8% | n/a | n/a | n/a | one rotation recorded (NOUL questions are not rotated); flips not measurable |
| Qwen3-4B | 519 merged rows | tool | 8 | 3 | 36.3% | 25.1% | 64.5 / 64.5 / 63.3 | 68.3% | 68.3% | 74.9% | 93 / 94 | 16.7% vs 18.4% (12.5%) |
| Qwen3-4B | 519 merged rows | command | 24 | 3 | 29.3% | 22.4% | 60.2 / 57.1 / 60.2 | 60.2% | 59.8% | 68.3% | 75 / 76 | 5.7% vs 1.3% (4.2%) |
| Qwen3-4B | 519 merged rows | is_write | 2 | 1 | n/a | n/a | n/a | 85.7% | n/a | n/a | n/a | one rotation recorded (NOUL questions are not rotated); flips not measurable |
| Qwen3-4B | 519 merged rows | touches_poe_power | 2 | 1 | n/a | n/a | n/a | 95.8% | n/a | n/a | n/a | one rotation recorded (NOUL questions are not rotated); flips not measurable |
| Qwen3.5-0.8B (thinking off) | 260 base rows | tool | 8 | 3 | 95.0% | 78.5% | 68.8 / 43.8 / 19.2 | 74.6% | 70.8% | 87.3% | 15 / 247 | 15.5% vs 33.3% (12.5%) |
| Qwen3.5-0.8B (thinking off) | 260 base rows | command | 24 | 3 | 85.8% | 71.0% | 35.8 / 26.2 / 23.5 | 35.0% | 35.4% | 45.0% | 3 / 223 | 12.1% vs 7.2% (4.2%) |
| Qwen3.5-0.8B (thinking off) | 260 base rows | is_write | 2 | 1 | n/a | n/a | n/a | 81.9% | n/a | n/a | n/a | one rotation recorded (NOUL questions are not rotated); flips not measurable |
| Qwen3.5-0.8B (thinking off) | 260 base rows | touches_poe_power | 2 | 1 | n/a | n/a | n/a | 80.8% | n/a | n/a | n/a | one rotation recorded (NOUL questions are not rotated); flips not measurable |

## Breakdowns (choice questions with R = 3)

| model | rows | question | group | n | flip rate | averaged acc | single-rotation acc (mean) | confident flips |
|---|---|---|---|---:|---:|---:|---:|---:|
| Qwen3-4B | 259 authored rows | tool | style=A | 93 | 43.0% | 58.1% | 53.4% | 39 |
| Qwen3-4B | 259 authored rows | tool | style=B | 24 | 33.3% | 79.2% | 73.6% | 8 |
| Qwen3-4B | 259 authored rows | tool | style=C | 28 | 25.0% | 60.7% | 59.5% | 7 |
| Qwen3-4B | 259 authored rows | tool | style=T | 25 | 44.0% | 48.0% | 46.7% | 11 |
| Qwen3-4B | 259 authored rows | tool | style=X | 89 | 31.5% | 84.3% | 79.0% | 28 |
| Qwen3-4B | 259 authored rows | tool | split=adversarial_confusables | 81 | 30.9% | 88.9% | 83.1% | 25 |
| Qwen3-4B | 259 authored rows | tool | split=intents_diagnose | 54 | 33.3% | 72.2% | 67.9% | 18 |
| Qwen3-4B | 259 authored rows | tool | split=intents_docs_and_oos | 76 | 53.9% | 34.2% | 29.8% | 40 |
| Qwen3-4B | 259 authored rows | tool | split=intents_other_tools | 48 | 20.8% | 83.3% | 81.9% | 10 |
| Qwen3-4B | 259 authored rows | command | style=A | 93 | 40.9% | 53.8% | 51.3% | 38 |
| Qwen3-4B | 259 authored rows | command | style=B | 24 | 12.5% | 79.2% | 75.0% | 3 |
| Qwen3-4B | 259 authored rows | command | style=C | 28 | 32.1% | 46.4% | 46.4% | 9 |
| Qwen3-4B | 259 authored rows | command | style=T | 25 | 28.0% | 56.0% | 58.7% | 7 |
| Qwen3-4B | 259 authored rows | command | style=X | 89 | 21.3% | 67.4% | 67.4% | 18 |
| Qwen3-4B | 259 authored rows | command | split=adversarial_confusables | 81 | 21.0% | 69.1% | 69.1% | 16 |
| Qwen3-4B | 259 authored rows | command | split=intents_diagnose | 54 | 53.7% | 40.7% | 37.7% | 29 |
| Qwen3-4B | 259 authored rows | command | split=intents_docs_and_oos | 76 | 27.6% | 60.5% | 61.0% | 21 |
| Qwen3-4B | 259 authored rows | command | split=intents_other_tools | 48 | 18.8% | 66.7% | 63.9% | 9 |
| Qwen3-4B | 519 merged rows | tool | style=A | 93 | 43.0% | 58.1% | 53.4% | 39 |
| Qwen3-4B | 519 merged rows | tool | style=B | 24 | 33.3% | 79.2% | 73.6% | 8 |
| Qwen3-4B | 519 merged rows | tool | style=C | 28 | 25.0% | 60.7% | 59.5% | 7 |
| Qwen3-4B | 519 merged rows | tool | style=T | 25 | 44.0% | 48.0% | 46.7% | 11 |
| Qwen3-4B | 519 merged rows | tool | style=X | 89 | 31.5% | 84.3% | 79.0% | 28 |
| Qwen3-4B | 519 merged rows | tool | split=adversarial_confusables | 81 | 30.9% | 88.9% | 83.1% | 25 |
| Qwen3-4B | 519 merged rows | tool | split=intents_diagnose | 54 | 33.3% | 72.2% | 67.9% | 18 |
| Qwen3-4B | 519 merged rows | tool | split=intents_docs_and_oos | 76 | 53.9% | 34.2% | 29.8% | 40 |
| Qwen3-4B | 519 merged rows | tool | split=intents_other_tools | 48 | 20.8% | 83.3% | 81.9% | 10 |
| Qwen3-4B | 519 merged rows | command | style=A | 93 | 40.9% | 53.8% | 51.3% | 38 |
| Qwen3-4B | 519 merged rows | command | style=B | 24 | 12.5% | 79.2% | 75.0% | 3 |
| Qwen3-4B | 519 merged rows | command | style=C | 28 | 32.1% | 46.4% | 46.4% | 9 |
| Qwen3-4B | 519 merged rows | command | style=T | 25 | 28.0% | 56.0% | 58.7% | 7 |
| Qwen3-4B | 519 merged rows | command | style=X | 89 | 21.3% | 67.4% | 67.4% | 18 |
| Qwen3-4B | 519 merged rows | command | split=adversarial_confusables | 81 | 21.0% | 69.1% | 69.1% | 16 |
| Qwen3-4B | 519 merged rows | command | split=intents_diagnose | 54 | 53.7% | 40.7% | 37.7% | 29 |
| Qwen3-4B | 519 merged rows | command | split=intents_docs_and_oos | 76 | 27.6% | 60.5% | 61.0% | 21 |
| Qwen3-4B | 519 merged rows | command | split=intents_other_tools | 48 | 18.8% | 66.7% | 63.9% | 9 |
| Qwen3.5-0.8B (thinking off) | 260 base rows | tool | style=A | 65 | 100.0% | 61.5% | 36.9% | 2 |
| Qwen3.5-0.8B (thinking off) | 260 base rows | tool | style=B | 65 | 86.2% | 81.5% | 50.8% | 6 |
| Qwen3.5-0.8B (thinking off) | 260 base rows | tool | style=C | 65 | 98.5% | 72.3% | 36.9% | 1 |
| Qwen3.5-0.8B (thinking off) | 260 base rows | tool | style=canon | 65 | 95.4% | 83.1% | 51.3% | 6 |
| Qwen3.5-0.8B (thinking off) | 260 base rows | tool | split=canonical | 65 | 95.4% | 83.1% | 51.3% | 6 |
| Qwen3.5-0.8B (thinking off) | 260 base rows | tool | split=paraphrase | 195 | 94.9% | 71.8% | 41.5% | 9 |
| Qwen3.5-0.8B (thinking off) | 260 base rows | command | style=A | 65 | 84.6% | 26.2% | 22.1% | 0 |
| Qwen3.5-0.8B (thinking off) | 260 base rows | command | style=B | 65 | 90.8% | 43.1% | 34.9% | 3 |
| Qwen3.5-0.8B (thinking off) | 260 base rows | command | style=C | 65 | 81.5% | 27.7% | 23.1% | 0 |
| Qwen3.5-0.8B (thinking off) | 260 base rows | command | style=canon | 65 | 86.2% | 43.1% | 33.8% | 0 |
| Qwen3.5-0.8B (thinking off) | 260 base rows | command | split=canonical | 65 | 86.2% | 43.1% | 33.8% | 0 |
| Qwen3.5-0.8B (thinking off) | 260 base rows | command | split=paraphrase | 195 | 85.6% | 32.3% | 26.7% | 3 |

## Flips as an uncertainty signal

| model | rows | question | averaged acc on flipped rows | on stable rows | P(flip given wrong) | P(flip given right) | abstain on flips: coverage, acc | AUROC stable→correct | AUROC max p→correct | wrong > 0.9 per rotation | rescued by averaging | wrong > 0.9 after averaging |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---|---|---:|
| Qwen3-4B | 259 authored rows | tool | 51.1% | 78.2% | 56.1% | 27.1% | 63.7%, 78.2% | 0.645 | 0.736 | [79, 74, 79] | [15, 8, 14] | 32 |
| Qwen3-4B | 259 authored rows | command | 34.2% | 71.0% | 48.5% | 16.7% | 70.7%, 71.0% | 0.659 | 0.762 | [91, 91, 80] | [7, 9, 3] | 51 |
| Qwen3-4B | 519 merged rows | tool | 51.1% | 78.2% | 56.1% | 27.1% | 63.7%, 78.2% | 0.645 | 0.736 | [79, 74, 79] | [15, 8, 14] | 32 |
| Qwen3-4B | 519 merged rows | command | 34.2% | 71.0% | 48.5% | 16.7% | 70.7%, 71.0% | 0.659 | 0.762 | [91, 91, 80] | [7, 9, 3] | 51 |
| Qwen3.5-0.8B (thinking off) | 260 base rows | tool | 73.7% | 92.3% | 98.5% | 93.8% | 5.0%, 92.3% | 0.523 | 0.815 | [1, 0, 0] | [1, 0, 0] | 0 |
| Qwen3.5-0.8B (thinking off) | 260 base rows | command | 28.7% | 73.0% | 94.1% | 70.3% | 14.2%, 73.0% | 0.619 | 0.839 | [0, 0, 0] | [0, 0, 0] | 0 |

## Position shares (pooled over rotations x rows)

| model | rows | question | first position: won / gold | last position: won / gold | most-won position | max gap | total variation |
|---|---|---|---|---|---|---:|---:|
| Qwen3-4B | 259 authored rows | tool | 16.7% / 18.4% | 20.1% / 18.5% | B: 22.0% won, 19.6% gold | 3.0% | 8.9% |
| Qwen3-4B | 259 authored rows | command | 5.7% / 1.3% | 12.1% / 20.6% | V: 12.1% won, 21.4% gold | 9.3% | 27.4% |
| Qwen3-4B | 519 merged rows | tool | 16.7% / 18.4% | 20.1% / 18.5% | B: 22.0% won, 19.6% gold | 3.0% | 8.9% |
| Qwen3-4B | 519 merged rows | command | 5.7% / 1.3% | 12.1% / 20.6% | V: 12.1% won, 21.4% gold | 9.3% | 27.4% |
| Qwen3.5-0.8B (thinking off) | 260 base rows | tool | 15.5% / 33.3% | 13.3% / 33.3% | B: 56.2% won, 23.1% gold | 33.1% | 40.1% |
| Qwen3.5-0.8B (thinking off) | 260 base rows | command | 12.1% / 7.2% | 0.1% / 5.6% | O: 20.0% won, 9.2% gold | 16.8% | 52.2% |

## Top flip pairs

- Qwen3-4B, 259 authored rows, tool: query_device_info ↔ configure_device_setting (33); query_device_info ↔ diagnose_port_issues (19); configure_device_setting ↔ diagnose_port_issues (12); query_device_info ↔ get_network_events (10); configure_device_setting ↔ search_documentation (9); configure_device_setting ↔ none_of_these (6). Flipped rows involving none_of_these: 17; involving a dangerous pair: 0.
- Qwen3-4B, 259 authored rows, command: show_port_status ↔ show_all_ports_status (14); show_cable_diagnostics ↔ none_of_these (7); enable_port ↔ disable_port (6); show_all_ports_status ↔ show_poe_status (6); configure_port_speed ↔ configure_port_duplex (5); show_poe_status ↔ show_all_poe_status (5). Flipped rows involving none_of_these: 21; involving a dangerous pair: 5.
- Qwen3-4B, 519 merged rows, tool: query_device_info ↔ configure_device_setting (33); query_device_info ↔ diagnose_port_issues (19); configure_device_setting ↔ diagnose_port_issues (12); query_device_info ↔ get_network_events (10); configure_device_setting ↔ search_documentation (9); configure_device_setting ↔ none_of_these (6). Flipped rows involving none_of_these: 17; involving a dangerous pair: 0.
- Qwen3-4B, 519 merged rows, command: show_port_status ↔ show_all_ports_status (14); show_cable_diagnostics ↔ none_of_these (7); enable_port ↔ disable_port (6); show_all_ports_status ↔ show_poe_status (6); configure_port_speed ↔ configure_port_duplex (5); show_poe_status ↔ show_all_poe_status (5). Flipped rows involving none_of_these: 21; involving a dangerous pair: 5.
- Qwen3.5-0.8B (thinking off), 260 base rows, tool: configure_device_setting ↔ diagnose_port_issues (117); query_device_info ↔ configure_device_setting (104); configure_device_setting ↔ search_documentation (90); diagnose_port_issues ↔ search_documentation (61); configure_device_setting ↔ calibrate_cable_baseline (29); query_device_info ↔ diagnose_port_issues (16). Flipped rows involving none_of_these: 0; involving a dangerous pair: 0.
- Qwen3.5-0.8B (thinking off), 260 base rows, command: show_all_ports_status ↔ show_poe_status (34); show_all_poe_status ↔ show_poe_config (33); show_poe_config ↔ configure_poe_mode (27); show_poe_status ↔ show_poe_config (22); show_poe_status ↔ configure_poe_mode (22); show_poe_status ↔ show_all_poe_status (21). Flipped rows involving none_of_these: 0; involving a dangerous pair: 31.
