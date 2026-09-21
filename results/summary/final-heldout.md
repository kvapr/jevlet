| backend | question | n | acc | acc (alt ok) | ECE raw | ECE T-scaled | Brier raw | wrong@conf>0.9 raw→T | flip rate | median latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ft-modernbert-base-desc-shuf-e8 | tool | 126 | 0.881 | nan | 0.047 | 0.021 | 0.172 | 1→2 | nan | 11 ms |
| ft-modernbert-base-desc-shuf-e8 | command | 126 | 0.778 | 0.794 | 0.069 | 0.058 | 0.300 | 1→3 | nan | 11 ms |
| ft-modernbert-base-desc-shuf-e8 | is_write | 126 | 0.952 | nan | 0.054 | 0.028 | 0.103 | 0→2 | nan | 11 ms |
| ft-modernbert-base-desc-shuf-e8 | touches_poe_power | 126 | 0.968 | nan | 0.088 | 0.007 | 0.076 | 0→3 | nan | 11 ms |
| ft-modernbert-base-desc-shuf-e8 | gate | 217 | AUROC 0.530 | acc@.5 0.544→0.544 | 0.063 | 0.012 | 0.255 | real misses flagged 5/5 | dangerous accepted 11/38 | 9 ms |
| llamacpp-readout | gate | 217 | AUROC 0.967 | acc@.5 0.848→0.848 | 0.145 | 0.075 | 0.143 | real misses flagged 3/5 | dangerous accepted 0/38 | 972 ms |
| llamacpp-readout | tool | 126 | 0.849 | nan | 0.088 | 0.047 | 0.281 | 8→2 | nan | 2163 ms |
| llamacpp-readout | command | 126 | 0.698 | 0.730 | 0.208 | 0.065 | 0.515 | 23→0 | nan | 1971 ms |
| llamacpp-readout | is_write | 126 | 0.905 | nan | 0.099 | 0.049 | 0.192 | 11→3 | nan | 466 ms |
| llamacpp-readout | touches_poe_power | 126 | 0.960 | nan | 0.042 | 0.017 | 0.080 | 5→2 | nan | 408 ms |
