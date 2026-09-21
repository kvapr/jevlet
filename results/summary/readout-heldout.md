| backend | question | n | acc | acc (alt ok) | ECE raw | ECE T-scaled | Brier raw | wrong@conf>0.9 raw→T | flip rate | median latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llamacpp-readout | tool | 126 | 0.849 | nan | 0.088 | 0.047 | 0.281 | 8→2 | nan | 2163 ms |
| llamacpp-readout | command | 126 | 0.698 | 0.730 | 0.208 | 0.065 | 0.515 | 23→0 | nan | 1971 ms |
| llamacpp-readout | is_write | 126 | 0.905 | nan | 0.099 | 0.049 | 0.192 | 11→3 | nan | 466 ms |
| llamacpp-readout | touches_poe_power | 126 | 0.960 | nan | 0.042 | 0.017 | 0.080 | 5→2 | nan | 408 ms |
| llamacpp-readout | gate | 217 | AUROC 0.967 | acc@.5 0.848→0.848 | 0.145 | 0.075 | 0.143 | real misses flagged 3/5 | dangerous accepted 0/38 | 972 ms |
