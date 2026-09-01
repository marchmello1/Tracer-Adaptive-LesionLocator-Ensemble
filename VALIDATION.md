# Paired interaction validation

The final policy was selected by replaying the same cumulative simulated
scribbles through the reference runtime and the final topology-safe policy. These
fixtures are local validation cases, not the hidden preliminary test set, so
the results demonstrate the direction of the changes rather than an expected
official score.

| Case | v2 Dice AUC | v3 Dice AUC | v2 F1 AUC | v3 F1 AUC |
| --- | ---: | ---: | ---: | ---: |
| FDG 1 | 0.859062 | 0.859103 | 0.750000 | 0.750000 |
| FDG 2 | 0.854681 | 0.854681 | 1.000000 | 1.000000 |
| FDG 3 (negative) | n/a | n/a | n/a | n/a |
| FDG 4 | 0.865234 | 0.865234 | 1.000000 | 1.000000 |
| FDG 5 | 0.757311 | 0.757798 | 0.774194 | 0.774194 |
| FDG 6 | 0.948866 | 0.948866 | 1.000000 | 1.000000 |
| PSMA 1 | 0.801071 | 0.801398 | 0.887139 | 0.889764 |
| PSMA 2 | 0.834320 | 0.836070 | 0.950188 | 0.950188 |
| PSMA 3 | 0.793657 | 0.823770 | 1.000000 | 1.000000 |
| PSMA 4 | 0.848048 | 0.851511 | 0.861111 | 0.861111 |
| Mean over 9 defined cases | 0.840250 | 0.844270 | 0.913626 | 0.913917 |

Relative to the frozen preliminary v2 image, v3 changed the local mean by
+0.004020 Dice AUC and +0.000292 F1 AUC. Five cases improved in Dice and four
were unchanged; one case improved in F1 and eight were unchanged. No defined case
regressed. V3 adds only exact foreground-scribble voxels, without dilation, and
rejects a connected scribble stroke if it would merge accepted lesions. These
fixtures support a causal engineering comparison but remain too small to
estimate hidden final-test performance.

Regression coverage includes independent donor-component topology validation,
rejection of bridges between existing lesions, safe acceptance of a separate
new lesion in the same interaction, the six-point PSMA evidence threshold, the
128-component burden ceiling, and prevention of background-induced component
splits.
