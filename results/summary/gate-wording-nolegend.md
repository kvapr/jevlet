| backend | question | n | acc | acc (alt ok) | ECE raw | ECE T-scaled | Brier raw | wrong@conf>0.9 raw→T | flip rate | median latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llamacpp-readout-wording-current | gate | 200 | AUROC 0.955 | acc@.5 0.825→0.825 | 0.165 | 0.081 | 0.166 | real misses flagged 4/7 | dangerous accepted 1/33 | 861 ms |
| llamacpp-readout-wording-earlier | gate | 200 | AUROC 0.953 | acc@.5 0.760→0.760 | 0.228 | 0.081 | 0.229 | real misses flagged 4/7 | dangerous accepted 0/33 | 855 ms |
