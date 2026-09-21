| backend | question | n | acc | acc (alt ok) | ECE raw | ECE T-scaled | Brier raw | wrong@conf>0.9 raw→T | flip rate | median latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| openjev-verdict-151m-onnx | tool | 260 | 0.727 | nan | 0.385 | 0.068 | 0.586 | 0→1 | nan | 385 ms |
| openjev-verdict-151m-onnx | command | 260 | 0.000 | 0.000 | 0.394 | 0.044 | 1.130 | 0→0 | nan | 385 ms |
| openjev-verdict-151m-onnx | is_write | 260 | 0.692 | nan | 0.060 | 0.008 | 0.434 | 0→0 | nan | 385 ms |
| openjev-verdict-151m-onnx | touches_poe_power | 260 | 0.215 | nan | 0.379 | 0.287 | 0.629 | 0→0 | nan | 385 ms |
| com-kotobalabs_open-jev-deberta-v3-large | tool | 260 | 0.238 | nan | 0.055 | 0.056 | 0.833 | 0→0 | nan | 53 ms |
| com-kotobalabs_open-jev-deberta-v3-large | command | 260 | 0.708 | 0.708 | 0.265 | 0.084 | 0.517 | 0→4 | nan | 45 ms |
| com-kotobalabs_open-jev-deberta-v3-large | is_write | 260 | 0.888 | nan | 0.088 | 0.037 | 0.162 | 0→2 | nan | 24 ms |
| com-kotobalabs_open-jev-deberta-v3-large | touches_poe_power | 260 | 0.385 | nan | 0.321 | 0.124 | 0.651 | 0→0 | nan | 25 ms |
| nli-cross-encoder_nli-deberta-v3-base | tool | 260 | 0.258 | nan | 0.054 | 0.131 | 0.950 | 2→0 | nan | 24 ms |
| nli-cross-encoder_nli-deberta-v3-base | command | 260 | 0.531 | 0.531 | 0.050 | 0.035 | 0.566 | 1→3 | nan | 42 ms |
| nli-cross-encoder_nli-deberta-v3-base | is_write | 260 | 0.865 | nan | 0.082 | 0.095 | 0.248 | 4→3 | nan | 8 ms |
| nli-cross-encoder_nli-deberta-v3-base | touches_poe_power | 260 | 0.754 | nan | 0.045 | 0.032 | 0.317 | 0→0 | nan | 8 ms |
| openjev-verdict-151m-onnx | gate | 939 | AUROC 0.566 | acc@.5 0.438→0.438 | 0.295 | 0.067 | 0.332 | real misses flagged 0/44 | dangerous accepted 156/156 | 141 ms |
| com-kotobalabs_open-jev-deberta-v3-large | gate | 939 | AUROC 0.663 | acc@.5 0.440→0.440 | 0.413 | 0.070 | 0.403 | real misses flagged 0/44 | dangerous accepted 155/156 | 27 ms |
| nli-cross-encoder_nli-deberta-v3-base | gate | 939 | AUROC 0.547 | acc@.5 0.442→0.442 | 0.401 | 0.069 | 0.420 | real misses flagged 1/44 | dangerous accepted 144/156 | 9 ms |
