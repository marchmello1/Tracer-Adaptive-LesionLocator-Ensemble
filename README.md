# Topology-Safe Adaptive PET/CT Lesion Fusion

This repository contains the Apache-2.0 AutoPET V topology-safe adaptive
submission. Its substantive changes are an independent per-component topology gate for
multi-click EDT fusion, topology-safe deletion of explicitly marked PSMA
background strokes, and activation of six-point EDT corrections for moderately
high-burden PSMA scans. Inference click coordinates are mapped into cropped
model space by subtracting the crop origin exactly once before resampling.
Exact foreground-scribble voxels are enforced without dilation when they do
not bridge accepted lesion components.
The upstream implementation validated all clicked
donor components together, allowing a harmful lesion bridge to be offset by a
separate newly created component, and disabled PSMA corrections above ten K0
components. For PSMA, the cutoff is raised to a conservative ceiling of 128 K0
components, allowing useful corrections at lesion burdens seen in validation
while protecting extremely fragmented scans. This decision remains stable
during stateless replay. This derivative accepts or rejects each clicked donor component
separately, preserving new lesions while rejecting bridges that reduce the
number of accepted components. See
`DERIVATION_NOTICE.md` and the regression test in
`tests/test_edt_stateless_fusion.py`.

## Base method

The method combines a five-fold AutoPET-III LesionTracer initial prediction with
tracer-specific K0 calibration and stateless EDT click correction.

## Method

- A fixed three-model router classifies FDG versus PSMA from PET/CT.
- K0 uses the five-fold AutoPET-III LesionTracer model with mirror TTA disabled.
- FDG uses probability threshold `0.47`; high-burden scans (at least 25 robust
  components) relax connected-component dust from 25 to 5 voxels.
- PSMA uses threshold `0.50`, dust 5, and a frozen logistic component pruner at
  false-positive threshold `0.86`.
- Interactive calls run the AutoPET-IV LesionLocator EDT fold-0 checkpoint.
  Click coordinates are transformed from the input grid into cropped model
  space with a single crop-origin subtraction before resampling.
  Tumor corrections are component-local and reject lesion bridges. Background
  deletion is disabled. PSMA correction requires at least six cumulative tumor
  points and at most 128 K0 components. Explicitly annotated PSMA background
  voxels are removed only when doing so cannot split a component. Exact tumor
  scribble voxels are added one connected stroke at a time; strokes that would
  merge accepted lesions are rejected.
- Each invocation is stateless: K0 is reconstructed and all cumulative clicks
  are replayed.

The Grand Challenge interface is:

```text
/input/images/ct/<case>.mha
/input/images/pet/<case>.mha
/input/lesion-clicks.json
/output/images/tumor-lesion-segmentation/<case>.mha
```

## Reproducible build

The full upstream Docker build downloads the public champion weights from
Zenodo and the EDT checkpoint from this repository's `weights-v1.0.0` GitHub
Release. `Dockerfile.slim`, used for the Grand Challenge upload, instead reads
the same hash-pinned checkpoints from the platform's separate model archive.

```bash
docker build --platform=linux/amd64 \
  --build-arg EDT_WEIGHTS_URL=https://github.com/Marchmello01/autopet-v-topology-safe-adaptive/releases/download/weights-v1.0.0/checkpoint_final.pth \
  -t autopet-v-toposafe-adaptive:final .
```

Frozen EDT checkpoint SHA-256:

```text
a0cb3a89c72b0a79a27900980361385ff02572c0c71aba6609390fecbbc13e82
```

No network access is used during inference. In the submitted deployment the
weights are mounted read-only at `/opt/ml/model` by Grand Challenge.

## GitHub authentication for maintainers

Never place a personal access token in this repository, a commit, a command-line
argument, or an issue. Use a fine-grained token restricted to this repository
with `Contents: Read and write` and `Metadata: Read-only`. Enter it without
echoing it and keep it only in the current shell:

```bash
read -rsp 'GitHub token: ' GH_TOKEN
export GH_TOKEN
printf '\n'
gh auth status
```

Revoke and replace any token disclosed in chat, logs, screenshots, or shell
history before using it.

## Model checkpoints

| Component | Source | Integrity check |
| --- | --- | --- |
| AutoPET-III LesionTracer, folds 0–4 | [Zenodo 14007247](https://zenodo.org/records/14007247) | MD5 `566016409b0bd14770c0b57c1f2873f1` |
| LesionLocator EDT, fold 0 | GitHub Release `weights-v1.0.0` | SHA-256 `a0cb3a89c72b0a79a27900980361385ff02572c0c71aba6609390fecbbc13e82` |

The 820 MB EDT checkpoint is intentionally not committed to Git. GitHub's
per-file source limit is 100 MB, so the checkpoint is distributed as a
versioned Release asset and verified during the Docker build.

## Repository layout

```text
candidate_runtime/       Final K0, PSMA pruning, and stateless fusion
autoPET-interactive/     Pinned EDT/nnU-Net fork (Apache-2.0)
champion/                Pinned AutoPET-III inference fork (Apache-2.0)
weights/edt_model/       EDT plans, metadata, and expected checkpoint hash
tests/                   Runtime and safety-gate unit tests
Dockerfile               Reproducible, digest-pinned container build
edt_runner.py             Isolated fold-0 EDT inference process
public_tracer_router.py   Fixed FDG/PSMA router
```

## Tests

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q tests
```

The tests cover invalid inputs, PSMA component pruning, cumulative-click
activation, topology-preserving donor fusion, exact supervised foreground
strokes, and safe background erasure.

## Validation summary

The frozen submission obtained Dice `0.854649`, lesion F1/DMM `0.834179`, and
mean position `3.5` on the AutoPET V Preliminary Test Set evaluation created
on 1 September 2026. This five-case implementation check is not an estimate of
performance on the 200-case multicenter final test set. Local paired replay was
used for interaction-policy selection; complete case-level AUC results and the
unchanged comparator are reported in `VALIDATION.md`. The official preliminary
evaluation identifier and recorded metric ranks are in
`docs/PRELIMINARY_RESULTS.md`.

The subsequent v3 foreground-stroke rule was evaluated causally on the same
local trajectories. Relative to v2 it increased mean Dice AUC from `0.840250`
to `0.844270` and mean F1 AUC from `0.913626` to `0.913917` over nine defined
cases after two additional FDG fixtures were added. Five cases improved in
Dice and four were unchanged; no per-case regression was observed. V3 has not
been evaluated on the official test set.

## Provenance

- AutoPET V public implementation: bundled Apache-2.0 source snapshot.
- AutoPET-III LesionTracer source and weights: bundled upstream code and Zenodo
  record 14007247.
- AutoPET interactive/EDT source: `MIC-DKFZ/autoPET-interactive`, pinned commit
  `0da0e7f`.

See `NOTICE` for attribution and modification notes.

## Citation

The complete LNCS method-description source and compiled manuscript are in
`paper/`. A public preprint identifier will be added after upload. Until then,
cite this repository URL and the upstream AutoPET-III and AutoPET interactive
projects listed above when reusing their components.

## License

Apache License 2.0. Bundled third-party source retains its upstream notices.
