| backend | question | n | acc | acc (alt ok) | ECE raw | ECE T-scaled | Brier raw | wrong@conf>0.9 raw→T | flip rate | median latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llamacpp-readout | gate | 939 | AUROC 0.943 | acc@.5 0.875→0.875 | 0.115 | 0.031 | 0.117 | real misses flagged 24/44 | dangerous accepted 1/156 | 985 ms |
| openjev-verdict-151m-onnx-short | tool | 260 | 0.585 | nan | 0.152 | 0.117 | 0.555 | 0→0 | nan | 385 ms |
| openjev-verdict-151m-onnx-short | command | 260 | 0.000 | 0.000 | 0.448 | 0.044 | 1.177 | 0→0 | nan | 385 ms |
| openjev-verdict-151m-onnx-short | is_write | 260 | 0.692 | nan | 0.060 | 0.008 | 0.434 | 0→0 | nan | 385 ms |
| openjev-verdict-151m-onnx-short | touches_poe_power | 260 | 0.215 | nan | 0.379 | 0.287 | 0.629 | 0→0 | nan | 385 ms |
| openjev-verdict-151m-onnx-short | gate | 939 | AUROC 0.566 | acc@.5 0.438→0.438 | 0.295 | 0.067 | 0.332 | real misses flagged 0/44 | dangerous accepted 156/156 | 139 ms |
