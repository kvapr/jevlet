| backend | question | n | acc | acc (alt ok) | ECE raw | ECE T-scaled | Brier raw | wrong@conf>0.9 raw→T | flip rate | median latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llamacpp-readout | gate | 939 | AUROC 0.952 | acc@.5 0.882→0.882 | 0.108 | 0.030 | 0.111 | real misses flagged 22/34 | dangerous accepted 1/156 | 985 ms |
| com-kotobalabs_open-jev-deberta-v3-large | gate | 939 | AUROC 0.668 | acc@.5 0.450→0.450 | 0.403 | 0.059 | 0.394 | real misses flagged 0/34 | dangerous accepted 155/156 | 27 ms |
