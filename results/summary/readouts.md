| backend | question | n | acc | acc (alt ok) | ECE raw | ECE T-scaled | Brier raw | wrong@conf>0.9 raw→T | flip rate | median latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llamacpp-readout | tool | 519 | 0.825 | nan | 0.092 | 0.059 | 0.299 | 35→8 | nan | 2033 ms |
| llamacpp-readout | command | 519 | 0.724 | 0.744 | 0.154 | 0.052 | 0.471 | 74→0 | nan | 1889 ms |
| llamacpp-readout | is_write | 519 | 0.919 | nan | 0.078 | 0.025 | 0.156 | 37→10 | nan | 451 ms |
| llamacpp-readout | touches_poe_power | 519 | 0.973 | nan | 0.026 | 0.005 | 0.052 | 12→8 | nan | 410 ms |
| llamacpp-readout | tool | 259 | 0.683 | nan | 0.179 | 0.054 | 0.532 | 31→0 | 0.36 | 3002 ms |
| llamacpp-readout | command | 259 | 0.602 | 0.641 | 0.261 | 0.041 | 0.655 | 51→0 | 0.29 | 2198 ms |
| llamacpp-readout | is_write | 259 | 0.857 | nan | 0.137 | 0.026 | 0.275 | 33→0 | 0.00 | 454 ms |
| llamacpp-readout | touches_poe_power | 259 | 0.958 | nan | 0.041 | 0.006 | 0.080 | 9→6 | 0.00 | 382 ms |
| llamacpp-readout | tool | 260 | 0.308 | nan | 0.183 | 0.183 | 0.875 | 0→0 | 0.00 | 1825 ms |
| llamacpp-readout | command | 260 | 0.000 | 0.000 | 0.333 | 0.047 | 1.313 | 0→0 | 1.00 | 1383 ms |
| llamacpp-readout | is_write | 260 | 0.692 | nan | 0.304 | 0.042 | 0.612 | 79→0 | 0.00 | 444 ms |
| llamacpp-readout | touches_poe_power | 260 | 0.219 | nan | 0.783 | 0.386 | 1.563 | 203→0 | 0.00 | 433 ms |
| llamacpp-readout | gate | 939 | AUROC 0.500 | acc@.5 0.448→0.448 | 0.552 | 0.154 | 0.552 | real misses flagged 0/34 | dangerous accepted 156/156 | 456 ms |
