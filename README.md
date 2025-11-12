Gaussian Splatting (Original + Vector-Quantized) — Monorepo
===========================================================

This repository is structured to implement the original 3D Gaussian Splatting method first, then add a Vector-Quantized (VQ) variant as a clean extension. It uses a modular package layout, a registry to swap implementations, and YAML configs to ensure experiment reproducibility.

Highlights
----------
- `gsplat/methods/`: Original and VQ implementations share a common interface.
- `configs/`: Method- and experiment-specific YAMLs (Hydra-compatible style).
- `apps/`: Small CLIs for training and rendering.
- `gsplat/render/`: Rendering backends (start simple, add CUDA later in `csrc/`).
- `tests/`: Sanity tests to keep structure healthy.

Quick Start
-----------
1) Install (editable mode):
```
pip install -e .
```

2) Train the original method on a dataset:
```
python -m apps.train --config configs/original.yaml
```

3) Train the VQ variant:
```
python -m apps.train --config configs/vq.yaml
```

4) Render from a checkpoint:
```
python -m apps.render --checkpoint runs/demo/checkpoint.pt --output outputs/demo/
```

Repository Layout
-----------------
```
gsplat/
  methods/
    base.py           # Abstract interface for all methods
    original/         # Original Gaussian Splatting
    vq/               # Vector-Quantized extension
  datasets/           # Dataset loaders (e.g., LLFF, NeRF-Synthetic)
  render/             # CPU/CUDA rendering backends
  optim/              # Optimizers/schedulers/regularizers
  utils/              # Logging, seeding, config helpers
  csrc/               # (Later) CUDA/C++ kernels + bindings
apps/
  train.py            # Training CLI
  render.py           # Rendering CLI
configs/
  original.yaml
  vq.yaml
tests/
  test_registry.py
docs/
  ARCHITECTURE.md
scripts/
  download_demo_data.sh
```

Incremental Plan
----------------
- Start with `methods/original` using pure Python/PyTorch and reference CPU rendering.
- Add CUDA kernels in `gsplat/csrc/` once correctness is established.
- Introduce `methods/vq` by subclassing the shared base, reusing data/renderer/optimizer where possible.
- Keep all experiment settings in `configs/` and log outputs to `runs/` and `outputs/`.

License
-------
TBD.


