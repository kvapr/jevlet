| backend | question | n | acc | acc (alt ok) | ECE raw | ECE T-scaled | Brier raw | wrong@conf>0.9 raw→T | flip rate | median latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llamacpp-readout-wording-current | gate | 400 | AUROC 0.957 | acc@.5 0.887→0.887 | 0.104 | 0.028 | 0.105 | real misses flagged 9/14 | dangerous accepted 0/70 | 849 ms |
| llamacpp-readout-wording-earlier | gate | 400 | AUROC 0.958 | acc@.5 0.850→0.850 | 0.140 | 0.053 | 0.140 | real misses flagged 9/14 | dangerous accepted 0/70 | 847 ms |
