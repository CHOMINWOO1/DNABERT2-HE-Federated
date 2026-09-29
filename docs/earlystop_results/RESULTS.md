# Historical early-stopping sensitivity analysis

H3K4me3, controlled non-IID 3-silo, seeds 42–46. Plain checkpoints were selected using DEV; HE used the same selected round count. This supplemental analysis does not replace the fixed 5-round primary study. Values were extracted from completed-run CSV files whose hashes match the original DONE records; no new training was performed.

| Seed | Plain FedAvg AUPRC | CKKS HE-FedAvg AUPRC | HE − Plain |
|---|---:|---:|---:|
| 42 | 0.77564880 | 0.76938499 | -0.00626381 |
| 43 | 0.76657914 | 0.76573562 | -0.00084352 |
| 44 | 0.76993773 | 0.77041144 | +0.00047371 |
| 45 | 0.76635454 | 0.76979132 | +0.00343678 |
| 46 | 0.76246272 | 0.76229882 | -0.00016390 |

Plain mean ± sample SD: 0.76819659 ± 0.00493593.
HE mean ± sample SD: 0.76752444 ± 0.00344391.
Paired difference: -0.00067215 ± 0.00352670.

See [protocol](protocol_KO.md), [aggregate JSON](summary.json), and [source hashes](provenance.json). The descriptive means alone do not establish non-inferiority. This is a synthetic silo split; model utility is not a cryptographic security proof.
