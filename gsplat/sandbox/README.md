# Sandbox: Training and Experiments

This directory contains experimental scripts and notebooks for training and testing 3D Gaussian Splatting.

## Files

### `train_fern.ipynb`
Interactive Jupyter notebook for training on the LLFF Fern dataset.

**Features:**
- Step-by-step walkthrough of the training process
- Visualizations of dataset and 3D point cloud
- Real-time rendering comparisons during training
- Loss and PSNR plots
- Gaussian statistics analysis

**Usage:**
```bash
# Start Jupyter from the sandbox directory
cd gsplat/sandbox
jupyter notebook train_fern.ipynb
```

Or run in VSCode with Jupyter extension.

### `colmap.py`
COLMAP visualization and debugging utilities (existing file).

---

## Quick Start: Training on Fern

### Option 1: Command Line (Python script)

```bash
# From project root
conda activate gausplats
python apps/train.py --config configs/original.yaml
```

The config file (`configs/original.yaml`) is already set up for the Fern dataset.

### Option 2: Interactive Notebook

```bash
# From project root
cd gsplat/sandbox
jupyter notebook train_fern.ipynb
```

Then run all cells sequentially.

---

## Configuration

Edit `configs/original.yaml` to change:
- Dataset paths (`data.colmap_path`, `data.images_path`)
- Training steps (`trainer.max_steps`)
- Learning rates (`lr_means`, `lr_rotations`, etc.)
- Gaussian initialization (`gaussian.initial_scale`, `gaussian.sh_degree`)
- Device (`device`: "cpu" or "cuda")

---

## Expected Results

With the current configuration (500 steps on Fern):
- **Initial PSNR**: ~5-10 dB (random colors from COLMAP initialization)
- **Expected PSNR after 500 steps**: ~15-20 dB (basic scene structure visible)
- **Training time (CPU)**: ~2-5 minutes for 500 steps
- **Training time (GPU)**: Not yet implemented (would be ~10-30 seconds)

**Note**: For production-quality results (>25 dB PSNR), you would need:
1. More training steps (10,000+)
2. Adaptive densification (clone/split/prune Gaussians)
3. CUDA acceleration
4. Higher-degree spherical harmonics (sh_degree > 0)

---

## Dataset Structure

The Fern dataset is already included at:
```
gsplat/datasets/nerf_llff_data/fern/
├── sparse/0/           # COLMAP reconstruction
│   ├── cameras.bin
│   ├── images.bin
│   └── points3D.bin
├── images_4/           # Downsampled images (4x)
│   ├── image000.png
│   ├── image001.png
│   └── ...
└── images/             # Original images (optional)
```

---

## Next Steps

1. **Run the notebook** to see the full training pipeline
2. **Experiment with hyperparameters** in the config file
3. **Try other datasets** by changing the paths in the config
4. **Implement densification** for better quality (future work)

---

## Troubleshooting

### Issue: "No module named 'gsplat'"
**Solution**: Make sure you're running from the project root or the path is added:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent.parent))
```

### Issue: Low PSNR / Bad quality
**Causes**:
- Not enough training steps (try 1000+)
- Learning rates too high/low (check for NaN losses)
- Need densification (current implementation has fixed Gaussians)

### Issue: Slow training
**Causes**:
- CPU-only implementation (GPU acceleration not yet implemented)
- Large images (use `images_4` or `images_8` for faster training)
- Too many Gaussians (current: ~25K for Fern)

---

## Performance Notes

Current implementation (CPU, unoptimized):
- ~4 FPS rendering (1000 Gaussians, 200×200)
- ~250ms per training step (includes backward pass)
- Memory: ~500MB for Fern dataset

For production use, CUDA kernels would provide 100-1000× speedup.

