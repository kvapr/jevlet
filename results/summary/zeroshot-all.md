| backend | question | n | acc | acc (alt ok) | ECE raw | ECE T-scaled | Brier raw | wrong@conf>0.9 raw→T | flip rate | median latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| laya-short | tool | 519 | 0.272 | nan | 0.333 | 0.123 | 0.932 | 13→0 | nan | 24 ms |
| laya-short | command | 519 | 0.468 | 0.468 | 0.526 | 0.017 | 1.054 | 268→0 | nan | 24 ms |
| laya-short | is_write | 519 | 0.867 | nan | 0.034 | 0.034 | 0.182 | 2→2 | nan | 24 ms |
| laya-short | touches_poe_power | 519 | 0.800 | nan | 0.020 | 0.024 | 0.284 | 2→1 | nan | 24 ms |
| laya-typed-decisions-short | tool | 519 | 0.617 | nan | 0.380 | 0.091 | 0.735 | 0→1 | nan | 28 ms |
| laya-typed-decisions-short | command | 519 | 0.514 | 0.538 | 0.322 | 0.097 | 0.790 | 91→0 | nan | 28 ms |
| laya-typed-decisions-short | is_write | 519 | 0.811 | nan | 0.093 | 0.043 | 0.300 | 0→2 | nan | 28 ms |
| laya-typed-decisions-short | touches_poe_power | 519 | 0.863 | nan | 0.158 | 0.039 | 0.264 | 0→5 | nan | 28 ms |
| com-kotobalabs_open-jev-deberta-v3-large | tool | 519 | 0.316 | nan | 0.120 | 0.081 | 0.814 | 0→0 | nan | 97 ms |
| com-kotobalabs_open-jev-deberta-v3-large | command | 519 | 0.451 | 0.468 | 0.082 | 0.077 | 0.724 | 1→3 | nan | 97 ms |
| com-kotobalabs_open-jev-deberta-v3-large | is_write | 519 | 0.753 | nan | 0.068 | 0.064 | 0.312 | 0→0 | nan | 97 ms |
| com-kotobalabs_open-jev-deberta-v3-large | touches_poe_power | 519 | 0.397 | nan | 0.304 | 0.111 | 0.642 | 0→0 | nan | 97 ms |
| nli-cross-encoder_nli-deberta-v3-base | tool | 519 | 0.206 | nan | 0.086 | 0.079 | 0.947 | 4→0 | nan | 57 ms |
| nli-cross-encoder_nli-deberta-v3-base | command | 519 | 0.408 | 0.418 | 0.057 | 0.048 | 0.724 | 4→2 | nan | 96 ms |
| nli-cross-encoder_nli-deberta-v3-base | is_write | 519 | 0.819 | nan | 0.034 | 0.059 | 0.271 | 8→6 | nan | 27 ms |
| nli-cross-encoder_nli-deberta-v3-base | touches_poe_power | 519 | 0.682 | nan | 0.113 | 0.043 | 0.420 | 5→0 | nan | 27 ms |
| laya-short | gate | 939 | AUROC 0.757 | acc@.5 0.592→0.592 | 0.178 | 0.022 | 0.270 | real misses flagged 7/34 | dangerous accepted 101/156 | 58 ms |
| laya-typed-decisions-short | gate | 939 | AUROC 0.734 | acc@.5 0.629→0.629 | 0.044 | 0.051 | 0.222 | real misses flagged 7/34 | dangerous accepted 54/156 | 58 ms |
| com-kotobalabs_open-jev-deberta-v3-large | gate | 939 | AUROC 0.668 | acc@.5 0.450→0.450 | 0.403 | 0.059 | 0.394 | real misses flagged 0/34 | dangerous accepted 155/156 | 80 ms |
| nli-cross-encoder_nli-deberta-v3-base | gate | 939 | AUROC 0.560 | acc@.5 0.453→0.453 | 0.390 | 0.058 | 0.410 | real misses flagged 1/34 | dangerous accepted 144/156 | 30 ms |
