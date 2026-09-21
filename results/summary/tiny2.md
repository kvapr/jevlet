| backend | question | n | acc | acc (alt ok) | ECE raw | ECE T-scaled | Brier raw | wrong@conf>0.9 raw→T | flip rate | median latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llamacpp-readout | tool | 260 | 0.746 | nan | 0.387 | 0.041 | 0.556 | 0→4 | 0.95 | 429 ms |
| llamacpp-readout | command | 260 | 0.350 | 0.350 | 0.108 | 0.145 | 0.782 | 0→0 | 0.86 | 451 ms |
| llamacpp-readout | is_write | 260 | 0.819 | nan | 0.160 | 0.016 | 0.314 | 0→2 | 0.00 | 142 ms |
| llamacpp-readout | touches_poe_power | 260 | 0.808 | nan | 0.123 | 0.095 | 0.224 | 0→0 | 0.00 | 143 ms |
| llamacpp-readout | gate | 939 | AUROC 0.879 | acc@.5 0.813→0.813 | 0.119 | 0.047 | 0.152 | real misses flagged 23/34 | dangerous accepted 25/156 | 317 ms |
