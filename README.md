# Tracer Adaptive LesionLocator Ensemble

An interactive PET/CT lesion-segmentation submission for AutoPET V. The
runtime produces an image-only lesion mask and then replays the accumulated
foreground and background annotations on every interaction step. It supports
both FDG and PSMA studies through one stateless Grand Challenge container.

Author: **Mohammad Agwan**, Nuvo AI. Contact:
**mohammad.agwan@nuvo.ai**.

## System at a glance

| Stage | Role |
| --- | --- |
| Tracer routing | A frozen three-model vote selects the FDG or PSMA inference policy. |
| Initial mask | Five LesionTracer folds generate the image-only prediction. |
| Calibration | Tracer-specific thresholds, dust removal, and a frozen PSMA component filter produce K0. |
| Interactive donor | A fold-0 EDT model predicts a correction from all accumulated annotations. |
| Reconciliation | Component-local rules accept useful corrections while preventing annotation-induced lesion merging or fragmentation. |

The network checkpoints remain fixed. This work concerns inference
orchestration, coordinate handling, tracer-aware calibration, and conservative
interactive reconciliation.

## Interaction algorithm

Every request is evaluated independently; the container does not rely on state
from an earlier invocation.

1. Validate CT, PET, and annotation inputs and preserve the original image
   geometry.
2. Recompute K0 from the PET/CT volume.
3. Normalize every cumulative annotation and transform it into cropped model
   space with exactly one crop-origin subtraction.
4. Run the EDT donor when the tracer-specific activation policy permits it.
5. Consider donor components separately. An addition is rejected if it joins
   previously distinct accepted lesions.
6. Apply supervised foreground voxels one connected stroke at a time. Apply a
   PSMA background removal only when it does not split an accepted component.
7. Resample the binary result to the source geometry and publish it through the
   Grand Challenge output contract.

The key design boundary is `TopologyLedger`: it is the sole owner of changes to
the accepted mask. Model execution, prompt normalization, policy selection,
and topology checks therefore remain independently testable.

```text
CT + PET ──> tracer router ──> five-fold K0 ──> calibration ───────┐
                                                                  │
annotations ──> validation ──> EDT donor ──> component proposals ─┤
                                                                  v
                                                        topology ledger
                                                                  │
                                                                  v
                                                        output segmentation
```

## Frozen inference policy

- FDG K0: probability threshold `0.47`; connected-component dust threshold 25
  voxels, relaxed to 5 for scans with at least 25 robust components.
- PSMA K0: probability threshold `0.50`; 5-voxel dust threshold; frozen
  logistic component rejection threshold `0.86`.
- Initial prediction: five LesionTracer folds with mirror TTA disabled.
- Interactive prediction: one LesionLocator EDT fold.
- PSMA EDT activation: at least six cumulative foreground points and no more
  than 128 K0 components.
- Foreground supervision is exact-voxel only; no unconditional dilation is
  performed.

## Challenge interface

```text
/input/images/ct/<case>.mha
/input/images/pet/<case>.mha
/input/lesion-clicks.json
/output/images/tumor-lesion-segmentation/<case>.mha
```

Inference is offline. In the submitted configuration, model data is mounted
read-only at `/opt/ml/model`.

## Source map

| Path | Responsibility |
| --- | --- |
| `candidate_runtime/challenge_io.py` | Input discovery, validation, and geometry-safe output |
| `candidate_runtime/initial_prediction.py` | K0 thresholds, filtering, and tracer calibration |
| `candidate_runtime/interactive_update.py` | EDT execution and correction orchestration |
| `candidate_runtime/edt_stateless_fusion.py` | Prompt representation, activation policy, and topology ledger |
| `candidate_runtime/psma_champion_pruner.py` | Frozen PSMA component classifier |
| `public_tracer_router.py` | Fixed FDG/PSMA routing ensemble |
| `tests/` | I/O, policy, topology, and regression tests |
| `docs/RUNTIME_REPRODUCTION.md` | Container-level GPU equivalence evidence |

## Model artifacts

| Artifact | Distribution | Integrity |
| --- | --- | --- |
| LesionTracer folds 0–4 | [Zenodo record 14007247](https://zenodo.org/records/14007247) | MD5 `566016409b0bd14770c0b57c1f2873f1` |
| LesionLocator EDT fold 0 | [Release `weights-v1.0.0`](https://github.com/marchmello1/Tracer-Adaptive-LesionLocator-Ensemble/releases/tag/weights-v1.0.0) | SHA-256 `a0cb3a89c72b0a79a27900980361385ff02572c0c71aba6609390fecbbc13e82` |

The 782 MiB EDT checkpoint is a release asset rather than a Git object. Verify
it before building:

```bash
sha256sum -c checkpoint_final.sha256
```

## Build and test

Install the lightweight test dependencies and run the focused suite:

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q tests
```

The release passes 45 focused tests. They cover invalid challenge inputs,
prompt normalization, K0 configuration, component filtering, cumulative-click
activation, topology-preserving additions and removals, exact foreground
strokes, idempotence, and release layout.

For a full image with model downloads:

```bash
docker build --platform=linux/amd64 \
  --build-arg EDT_WEIGHTS_URL=https://github.com/marchmello1/Tracer-Adaptive-LesionLocator-Ensemble/releases/download/weights-v1.0.0/checkpoint_final.pth \
  -t tracer-lesionlocator-ensemble:v1.0.0 .
```

`Dockerfile.slim` is the no-download deployment path used when Grand Challenge
mounts the separate model archive.

## Reproduction record

The modular runtime was compared with the frozen submission image using the
same container configuration, model mount, and inputs on an NVIDIA GPU. One
FDG fixture and one PSMA fixture produced byte-identical segmentation files.
Image digests, output hashes, commands, and scope limitations are recorded in
[`docs/RUNTIME_REPRODUCTION.md`](docs/RUNTIME_REPRODUCTION.md).

The preliminary evaluation created on 1 September 2026 reported:

| Metric | Value |
| --- | ---: |
| Dice | `0.854649` |
| Lesion F1/DMM | `0.834179` |
| Mean position | `3.5` |

This five-case preliminary evaluation is an implementation check, not a
guarantee of performance or rank on the multicenter final set. The evaluation
identifier and recorded ranks are preserved in
[`docs/PRELIMINARY_RESULTS.md`](docs/PRELIMINARY_RESULTS.md); local policy
experiments are documented in [`VALIDATION.md`](VALIDATION.md).

## Reuse and provenance

The repository includes pinned, permissively licensed upstream components from
the MIC-DKFZ AutoPET ecosystem. Their source notices and modification records
are preserved in [`NOTICE`](NOTICE) and
[`DERIVATION_NOTICE.md`](DERIVATION_NOTICE.md). The submission-specific runtime
is separated under `candidate_runtime/` so its behavior and changes can be
reviewed independently.

## Paper and citation

The method-description source is in [`paper/`](paper/), with the compiled
manuscript at
[`Tracer-Adaptive-LesionLocator-Ensemble.pdf`](paper/Tracer-Adaptive-LesionLocator-Ensemble.pdf).
The reviewed PDF and reproducible source package are published in the
[`preprint-v1.0.0` release](https://github.com/marchmello1/Tracer-Adaptive-LesionLocator-Ensemble/releases/tag/preprint-v1.0.0).
Citation metadata is available in [`CITATION.cff`](CITATION.cff).

## License

Apache License 2.0. Bundled third-party material retains its original notices.
