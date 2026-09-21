| backend | question | n | acc | acc (alt ok) | ECE raw | ECE T-scaled | Brier raw | wrong@conf>0.9 raw→T | flip rate | median latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ft-modernbert-base | tool | 126 | 0.706 | nan | 0.127 | 0.083 | 0.431 | 0→0 | nan | 28 ms |
| ft-modernbert-base | command | 126 | 0.651 | 0.659 | 0.095 | 0.067 | 0.489 | 0→1 | nan | 28 ms |
| ft-modernbert-base | is_write | 126 | 0.817 | nan | 0.134 | 0.073 | 0.328 | 0→2 | nan | 28 ms |
| ft-modernbert-base | touches_poe_power | 126 | 0.825 | nan | 0.129 | 0.015 | 0.297 | 0→0 | nan | 28 ms |
| ft-modernbert-base | gate | 217 | AUROC 0.647 | acc@.5 0.553→0.553 | 0.107 | 0.062 | 0.243 | real misses flagged 5/5 | dangerous accepted 0/38 | 38 ms |
