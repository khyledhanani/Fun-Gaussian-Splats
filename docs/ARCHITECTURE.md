Architecture Notes
==================

Goals
-----
- Clean separation between method logic, rendering backend, and data IO
- Extensible registry for swapping implementations
- Config-driven experiments
- COLMAP-first workflow (standard for Gaussian Splatting)

Key Modules
-----------
- `gsplat/methods/base.py`: Abstract base class for all methods
- `gsplat/methods/original/`: Original Gaussian Splatting method
- `gsplat/methods/vq/`: Vector-Quantized extension
- `gsplat/render/`: Rendering backends (CPU first, CUDA later)
- `gsplat/datasets/colmap_dataset.py`: Primary dataset loader for COLMAP outputs
  - Reads cameras.bin/txt, images.bin/txt, points3D.bin/txt
  - Provides images, camera poses, and initial 3D points for Gaussian initialization
- `gsplat/datasets/nerf_dataset.py`: Optional loader for NeRF-style datasets (testing)
- `gsplat/utils/`: Logging, seed, helpers

Data Flow
----------
1. **COLMAP Input**: Sparse reconstruction directory containing:
   - `cameras.bin` or `cameras.txt` (intrinsics: focal length, principal point)
   - `images.bin` or `images.txt` (extrinsics: poses + image filenames)
   - `points3D.bin` or `points3D.txt` (sparse 3D points with colors)

2. **Dataset Loader**: `ColmapDataset` parses COLMAP format and provides:
   - Image tensors + camera parameters per training sample
   - Initial 3D point cloud via `get_points3d()` method

3. **Gaussian Initialization**: Method converts COLMAP points → Gaussians:
   - Position: from points3D coordinates
   - Color: from points3D RGB
   - Scale/Rotation: initialized heuristically
   - Opacity: initialized uniformly

4. **Training**: Optimize Gaussians to match input images via differentiable rendering

Extending
---------
- Add a new method by subclassing `GaussianSplatMethod` and registering:
```
@METHOD_REGISTRY.register("my_method")
class MyMethod(GaussianSplatMethod):
    ...
```

- Add a new dataset by implementing the same interface as `ColmapDataset`:
  - `__init__`, `__len__`, `__iter__`, `get_points3d()` (if applicable)


