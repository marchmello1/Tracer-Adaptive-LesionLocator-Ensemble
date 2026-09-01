# Reproducing the submitted runtime

The public repository provides two supported paths.

## Self-contained build

```bash
git clone https://github.com/marchmello1/Tracer-Adaptive-LesionLocator-Ensemble.git
cd Tracer-Adaptive-LesionLocator-Ensemble
docker build --platform=linux/amd64 \
  -t tracer-lesionlocator-ensemble:v1.0.0 .
```

The build obtains the two large public model artifacts automatically:

| Runtime component | Public source | Build-time integrity check |
| --- | --- | --- |
| LesionTracer folds 0–4 | Zenodo record 14007247 | MD5 `566016409b0bd14770c0b57c1f2873f1` |
| LesionLocator EDT fold 0 | GitHub release `weights-v1.0.0` | SHA-256 `a0cb3a89c72b0a79a27900980361385ff02572c0c71aba6609390fecbbc13e82` |

The tracer router parameters, nnU-Net plans, runtime source, pinned dependency
versions, and upstream source snapshots are tracked directly in Git. The base
CUDA/PyTorch image is pinned by its content digest. A digest mismatch aborts
the image build.

## Grand Challenge model-mount build

`Dockerfile.slim` omits large checkpoints from the image and expects the
platform model archive at `/opt/ml/model`. The submitted archive has SHA-256
`5f092f510a61e40f11ca22307f923b7baa1783c5ac9c985250fe325400c48103`
and contains the same public checkpoints listed above. The 4.58 GB platform
archive is not duplicated as a GitHub release asset.

## Verification scope

The release has 45 focused tests. The modular fusion policy matched the frozen
policy in 1,000 randomized comparisons. With identical inputs and model mount,
the frozen and modular GPU containers produced byte-identical outputs on one
FDG and one PSMA fixture. The self-contained modular image was also rerun
without a host source mount on PSMA and retained the same output hash.

These checks verify the submitted runtime and the tested container artifact.
They do not constitute a clean-room build of every public dependency at an
arbitrary future date. Reproduction therefore also depends on the pinned base
image, Python package index, Zenodo record, and GitHub release remaining
available.
