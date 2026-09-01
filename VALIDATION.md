# Paired interaction validation

The final policy was selected by replaying the same cumulative simulated
scribbles through the reference runtime and the final topology-safe policy. These
fixtures are local validation cases, not the hidden preliminary test set, so
the results demonstrate the direction of the changes rather than an expected
official score.

| Case | Upstream Dice AUC | Final Dice AUC | Upstream F1 AUC | Final F1 AUC |
| --- | ---: | ---: | ---: | ---: |
| FDG 1 | 0.859062 | 0.859062 | 0.750000 | 0.750000 |
| FDG 2 | 0.854681 | 0.854681 | 1.000000 | 1.000000 |
| FDG 3 (negative) | n/a | n/a | n/a | n/a |
| FDG 4 | 0.865234 | 0.865234 | 1.000000 | 1.000000 |
| PSMA 1 | 0.800595 | 0.801071 | 0.887139 | 0.887139 |
| PSMA 2 | 0.827766 | 0.834320 | 0.929577 | 0.950188 |
| PSMA 3 | 0.793657 | 0.793657 | 1.000000 | 1.000000 |
| PSMA 4 | 0.848048 | 0.848048 | 0.861111 | 0.861111 |
| Mean over 7 defined cases | 0.835578 | 0.836582 | 0.918261 | 0.921205 |

The candidate therefore changed the local mean by +0.001004 Dice AUC and
+0.002944 F1 AUC. The PSMA 2 improvement came from recovering additional true
lesion instances without adding false-positive instances. The 128-component
ceiling kept the highly fragmented PSMA 1 scan on the conservative upstream
foreground path; only its topology-safe certified-background operation fired.

Regression coverage includes independent donor-component topology validation,
rejection of bridges between existing lesions, safe acceptance of a separate
new lesion in the same interaction, the six-point PSMA evidence threshold, the
128-component burden ceiling, and prevention of background-induced component
splits.
