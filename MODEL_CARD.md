# Model card

## Intended use

This research model produces and interactively refines binary tumor-lesion
segmentations from co-registered whole-body CT and FDG- or PSMA-PET. It accepts
cumulative foreground and background scribbles through the AutoPET V Grand
Challenge interface.

## Method

The initial segmentation is a five-fold 3-D residual-encoder nnU-Net ensemble.
A fixed image-derived router selects tracer-specific calibration. Interactive
updates map clicks into cropped model space using one crop-origin subtraction,
then use a fold-0 EDT model followed by per-component topology validation.
The submitted configuration is fully specified in `README.md`, and exact
checkpoint hashes are recorded in `WEIGHTS_MANIFEST.json`.

## Limitations

- This is a challenge research system, not a medical device.
- It has not been validated for autonomous diagnosis, treatment planning, or
  clinical decision-making.
- Performance may change across institutions, scanners, tracers, acquisition
  protocols, and disease distributions.
- The Preliminary Test Set contains only five cases and cannot establish
  final-test or clinical performance.
- Interaction quality depends on the location and polarity of supplied
  scribbles.

## Reproducibility

Inference is stateless: every invocation reconstructs the initial prediction
and replays all cumulative scribbles. Mirror test-time augmentation is disabled.
The Docker runtime uses no network access during inference. Tests cover input
validation, geometry preservation, tracer-dependent filtering, component
fusion, and topology guards.

## License and attribution

Repository code is released under Apache-2.0. Bundled third-party components
retain their original notices. Checkpoint redistribution remains subject to
the notices and terms supplied with the corresponding public releases.
