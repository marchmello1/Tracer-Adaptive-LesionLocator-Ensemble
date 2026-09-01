# Runtime reproduction audit

Date: 2026-09-01

The submission-specific runtime is separated into challenge-I/O,
initial-prediction, interactive-donor, and topology-reconciliation modules.
This structure is covered by 45 repository tests. With experimental behavior
disabled, the modular fusion policy also matched the frozen production policy
in 1,000/1,000 randomized comparisons.

## GPU container reproduction

The modular runtime archive with SHA-256
`7f7618f7973ddd76e2f96edc607a0e44751160371059db1cb5c9d20acf782c9d`
was mounted read-only over the custom runtime in flattened image
`eabcb6796fb2`. Baseline and modular runs used the same models, inputs,
environment, and 16 GB container shared-memory allocation.

| Fixture | Baseline output SHA-256 | Modular output SHA-256 | Result |
| --- | --- | --- | --- |
| FDG4 with cumulative foreground/background prompts | `c1fdf08cacf18bb9b3caae84ab3002474aa6fa99154cd8ef689b758dcc0532be` | `c1fdf08cacf18bb9b3caae84ab3002474aa6fa99154cd8ef689b758dcc0532be` | byte-identical |
| PSMA3 with foreground prompts | `46ed33a49d9a2a0c2de0e352fabca8bf5a80a00ac8a313426235bbbb1e33cd71` | `46ed33a49d9a2a0c2de0e352fabca8bf5a80a00ac8a313426235bbbb1e33cd71` | byte-identical |

The highest sampled GPU allocation was approximately 7.1 GB. This is evidence
for the tested fixtures and is not presented as a formal all-case peak-memory
bound.

The runtime was then embedded into a self-contained image and rerun without a
host code mount. Its PSMA3 output retained the exact hash above. The resulting
Docker-save archive is 7,400,811,008 bytes, contains `manifest.json`, and has
SHA-256
`30319b329450e27993fdce866ef1a8312bfa6a2b1c7bd9775c55643e01c8b4a3`.

## Scope

These checks establish packaging and behavioral equivalence on one FDG and one
PSMA interactive fixture. They do not estimate hidden-test rank or establish a
formal worst-case runtime bound.
