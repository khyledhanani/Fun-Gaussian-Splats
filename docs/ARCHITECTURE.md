Architecture Notes
==================

Goals
-----
- Clean separation between method logic, rendering backend, and data IO
- Extensible registry for swapping implementations
- Config-driven experiments

Key Modules
-----------
- `gsplat/methods/base.py`: Abstract base class for all methods
- `gsplat/methods/original/`: Original Gaussian Splatting method
- `gsplat/methods/vq/`: Vector-Quantized extension
- `gsplat/render/`: Rendering backends (CPU first, CUDA later)
- `gsplat/datasets/`: Dataset loaders for NeRF-style datasets
- `gsplat/utils/`: Logging, seed, helpers

Extending
---------
- Add a new method by subclassing `GaussianSplatMethod` and registering:
```
@METHOD_REGISTRY.register("my_method")
class MyMethod(GaussianSplatMethod):
    ...
```


