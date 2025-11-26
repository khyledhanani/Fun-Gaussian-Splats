Gaussian Splatting (Normal + Vector-Quantized) — Monorepo
===========================================================

This repository is structured to implement the original 3D Gaussian Splatting method first, then add a Vector-Quantized (VQ) variant as a clean extension. It uses a modular package layout, a registry to swap implementations, and YAML configs to ensure experiment reproducibility.

**Workflow**: Standard Gaussian Splatting pipeline starts with COLMAP reconstruction:
1. Run COLMAP SfM on input images → get camera poses + sparse point cloud
2. Initialize Gaussians from COLMAP 3D points
3. Train Gaussians to match input images
4. Render novel views

Highlights
----------
- `gsplat/methods/`: Original and VQ implementations share a common interface.
- `gsplat/datasets/colmap_dataset.py`: Primary dataset loader for COLMAP outputs.
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

2) Prepare COLMAP data:
   - Run COLMAP on your images to get sparse reconstruction
   - Or use a pre-processed COLMAP dataset

3) Train the original method:
```
python -m apps.train --config configs/original.yaml
```

4) Train the VQ variant:
```
python -m apps.train --config configs/vq.yaml
```

5) Render from a checkpoint:
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
  datasets/
    colmap_dataset.py # Primary: COLMAP sparse reconstruction loader
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
1. **Foundation**: COLMAP dataset loader + Gaussian representation
   - Parse COLMAP cameras/images/points3D (binary or text format)
   - Initialize Gaussians from COLMAP 3D points
   
2. **Rendering**: CPU reference implementation
   - Project 3D Gaussians to 2D, alpha blending
   - Validate correctness before optimization
   
3. **Training**: Full optimization loop
   - Loss functions, optimizer setup
   - Densification/pruning logic
   
4. **Optimization**: CUDA acceleration
   - Add CUDA kernels in `gsplat/csrc/` once correctness is established
   
5. **Extensions**: VQ variant
   - Introduce `methods/vq` by subclassing the shared base
   - Reuse data/renderer/optimizer where possible
   
6. **Polish**: Checkpointing, logging, visualization
   - Keep all experiment settings in `configs/`
   - Log outputs to `runs/` and `outputs/`

License
-------
TBD.


