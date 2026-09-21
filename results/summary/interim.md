| backend | question | n | acc | acc (alt ok) | ECE raw | ECE T-scaled | Brier raw | wrong@conf>0.9 raw→T | flip rate | median latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llamacpp-readout | tool | 260 | 0.965 | nan | 0.033 | 0.013 | 0.065 | 4→6 | nan | 1218 ms |
| llamacpp-readout | command | 260 | 0.846 | 0.846 | 0.133 | 0.070 | 0.287 | 23→12 | nan | 1487 ms |
| llamacpp-readout | is_write | 260 | 0.981 | nan | 0.019 | 0.008 | 0.038 | 4→1 | nan | 436 ms |
| llamacpp-readout | touches_poe_power | 260 | 0.988 | nan | 0.014 | 0.010 | 0.025 | 3→3 | nan | 438 ms |
| openjev-verdict-151m-onnx | tool | 260 | 0.727 | nan | 0.385 | 0.068 | 0.586 | 0→1 | nan | 385 ms |
| openjev-verdict-151m-onnx | command | 260 | 0.000 | 0.000 | 0.394 | 0.044 | 1.130 | 0→0 | nan | 385 ms |
| openjev-verdict-151m-onnx | is_write | 260 | 0.692 | nan | 0.060 | 0.008 | 0.434 | 0→0 | nan | 385 ms |
| openjev-verdict-151m-onnx | touches_poe_power | 260 | 0.215 | nan | 0.379 | 0.287 | 0.629 | 0→0 | nan | 385 ms |
