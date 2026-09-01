# Derivation notice

This deployment composes Apache-2.0 AutoPET-III LesionTracer and AutoPET-IV
LesionLocator components with submission-specific routing and topology-safe
interaction logic.

Changes made for the Marchmello01 topology-safe adaptive submission:

- resolve the unchanged AutoPET-III and EDT weights from Grand Challenge's
  separately mounted `/opt/ml/model` directory;
- add `Dockerfile.slim` to reuse an offline, validated CUDA/Python runtime;
- validate every clicked EDT donor component independently before fusion,
  preventing a harmful lesion bridge from being hidden by another donor
  component that creates a new lesion in the same interaction;
- add a regression test for that multi-click bridge/new-lesion failure mode;
- map inference clicks into cropped model space with one crop-origin
  subtraction before resampling, avoiding a second offset during clamping;
- retain the six-point PSMA evidence threshold while raising the upstream
  ten-component burden cutoff to a conservative 128-component ceiling,
  allowing stateless replay of topology-checked missed lesions while keeping
  extremely fragmented scans on the upstream conservative path;
- erase only explicitly annotated PSMA background-scribble strokes, accepting
  a stroke only when it cannot fragment a predicted component;
- retain the upstream `LICENSE` and `NOTICE` in the container.

The image thresholds, tracer router, K0 calibration, PSMA pruner, and EDT
inference are documented in the repository and method description. The bundled
DKFZ, nnU-Net, LesionTracer, and LesionLocator licenses and notices must be
retained in redistributions.
