# Camera Projection Guide for Gaussian Splatting

This guide explains where to find camera intrinsic and extrinsic information in the nerf_llff_data folder and how to use it to project 3D Gaussian Splats to 2D image space.

## Overview

Your nerf_llff_data folders contain COLMAP sparse reconstructions that include:
- **Camera Intrinsics** (K matrix): Stored in `sparse/0/cameras.bin`
- **Camera Extrinsics** (R, t): Stored in `sparse/0/images.bin`
- **3D Points**: Stored in `sparse/0/points3D.bin`

## File Locations

For each scene (e.g., `fern`, `flower`, etc.), the camera data is in:
```
gsplat/datasets/nerf_llff_data/{scene_name}/
├── sparse/0/
│   ├── cameras.bin    # Camera intrinsics (focal length, principal point, etc.)
│   ├── images.bin     # Camera extrinsics (pose for each image)
│   └── points3D.bin   # Sparse 3D point cloud
├── poses_bounds.npy   # Alternative format (NeRF LLFF)
└── images/            # Original images
```

## Camera Intrinsics (K Matrix)

### Reading Intrinsics
```python
from gsplat.datasets.colmap import read_cameras_binary

cameras = read_cameras_binary("sparse/0/cameras.bin")
cam = cameras[1]  # Usually only one camera model

# Camera properties
print(f"Model: {cam.model}")        # e.g., "SIMPLE_RADIAL", "PINHOLE"
print(f"Width: {cam.width}")        # Image width in pixels
print(f"Height: {cam.height}")      # Image height in pixels
print(f"Params: {cam.params}")      # Camera parameters
```

### Common Camera Models

**SIMPLE_RADIAL** (most common in nerf_llff_data):
```python
f, cx, cy, k = cam.params
# f: focal length (same for x and y)
# cx, cy: principal point (usually image center)
# k: radial distortion coefficient
```

**PINHOLE**:
```python
fx, fy, cx, cy = cam.params
# fx, fy: focal lengths for x and y axes
# cx, cy: principal point
```

**SIMPLE_PINHOLE**:
```python
f, cx, cy = cam.params
# f: focal length (same for x and y)
# cx, cy: principal point
```

### Constructing the K Matrix

For projection purposes (ignoring distortion):
```python
import numpy as np

# For SIMPLE_RADIAL or SIMPLE_PINHOLE
f, cx, cy = cam.params[:3]
K = np.array([
    [f,  0, cx],
    [0,  f, cy],
    [0,  0,  1]
])

# For PINHOLE
fx, fy, cx, cy = cam.params
K = np.array([
    [fx,  0, cx],
    [0,  fy, cy],
    [0,   0,  1]
])
```

Example from fern dataset:
```
K = [[3260.53,    0.00, 2016.00],
     [   0.00, 3260.53, 1512.00],
     [   0.00,    0.00,    1.00]]
```

## Camera Extrinsics (R, t)

### Reading Extrinsics
```python
from gsplat.datasets.colmap import read_images_binary

images = read_images_binary("sparse/0/images.bin")
img = images[1]  # Get pose for image with ID 1

# Image properties
print(f"Filename: {img.name}")          # e.g., "IMG_4026.JPG"
print(f"Camera ID: {img.camera_id}")    # Links to camera in cameras.bin
print(f"Quaternion: {img.qvec}")        # Rotation as quaternion [qw, qx, qy, qz]
print(f"Translation: {img.tvec}")       # Translation vector [tx, ty, tz]
```

### Converting to Rotation Matrix
```python
# Convert quaternion to 3x3 rotation matrix
R = img.qvec2rotmat()

# Translation vector
t = img.tvec

# Extrinsic matrix [R|t] (3x4)
extrinsic = np.column_stack([R, t])
```

Example from fern dataset (first image):
```
R = [[ 0.990, -0.027, -0.141],
     [ 0.022,  0.999, -0.035],
     [ 0.142,  0.032,  0.989]]

t = [3.558, 1.675, 0.847]

[R|t] = [[ 0.990, -0.027, -0.141,  3.558],
         [ 0.022,  0.999, -0.035,  1.675],
         [ 0.142,  0.032,  0.989,  0.847]]
```

## Projecting 3D Points to 2D

### Single Point Projection
```python
def project_3d_to_2d(point_3d, K, R, t):
    """
    Project a 3D point to 2D pixel coordinates.
    
    Args:
        point_3d: (3,) array with [X, Y, Z] world coordinates
        K: (3, 3) camera intrinsic matrix
        R: (3, 3) rotation matrix (world to camera)
        t: (3,) translation vector
    
    Returns:
        u, v: 2D pixel coordinates
        depth: depth in camera frame (positive = in front)
    """
    # Transform to camera coordinates
    point_cam = R @ point_3d + t
    
    # Project to image plane
    point_2d_homogeneous = K @ point_cam
    
    # Normalize by depth (perspective division)
    u = point_2d_homogeneous[0] / point_2d_homogeneous[2]
    v = point_2d_homogeneous[1] / point_2d_homogeneous[2]
    depth = point_2d_homogeneous[2]
    
    return u, v, depth
```

**Usage:**
```python
# Project a Gaussian center
gaussian_center = np.array([1.5, 2.3, 10.0])  # 3D world position
u, v, depth = project_3d_to_2d(gaussian_center, K, R, t)

# Check if visible
is_visible = (0 <= u < width and 0 <= v < height and depth > 0)
```

### Batch Projection (Efficient)
```python
def project_batch_3d_to_2d(points_3d, K, R, t, width, height):
    """
    Project multiple 3D points to 2D efficiently.
    
    Args:
        points_3d: (N, 3) array of 3D world points
        K: (3, 3) camera intrinsic matrix
        R: (3, 3) rotation matrix
        t: (3,) translation vector
        width, height: image dimensions
    
    Returns:
        points_2d: (N, 2) array of 2D pixel coordinates
        depths: (N,) array of depths
        valid_mask: (N,) boolean mask of visible points
    """
    # Transform to camera coordinates
    # (R @ X.T).T + t = X @ R.T + t (more efficient for many points)
    points_cam = points_3d @ R.T + t
    
    # Project to image plane
    points_2d_homogeneous = points_cam @ K.T
    
    # Extract depths
    depths = points_2d_homogeneous[:, 2]
    
    # Normalize by depth (avoid division by zero)
    points_2d = points_2d_homogeneous[:, :2] / (depths[:, np.newaxis] + 1e-10)
    
    # Visibility mask: inside bounds and in front of camera
    valid_mask = (
        (points_2d[:, 0] >= 0) & 
        (points_2d[:, 0] < width) &
        (points_2d[:, 1] >= 0) & 
        (points_2d[:, 1] < height) &
        (depths > 0)
    )
    
    return points_2d, depths, valid_mask
```

## Projecting Gaussian Covariance (3D to 2D)

For full Gaussian Splatting, you need to project both the **mean** (center) and **covariance** matrix.

### 3D Gaussian Representation
A 3D Gaussian is defined by:
- **μ**: 3D mean/center position (3,)
- **Σ**: 3D covariance matrix (3x3, symmetric positive definite)

### Projection Formula

The 2D covariance **Σ'** is computed as:
```
Σ' = J * Σ * J^T
```

Where **J** is the Jacobian of the projection function.

### Computing the Jacobian

For perspective projection `π: ℝ³ → ℝ²`:
```
π([x, y, z]) = [fx*x/z + cx, fy*y/z + cy]
```

The Jacobian at point `p_cam = [x, y, z]` in camera coordinates is:
```
J = [∂u/∂x  ∂u/∂y  ∂u/∂z]
    [∂v/∂x  ∂v/∂y  ∂v/∂z]

J = [fx/z    0     -fx*x/z²]
    [  0   fy/z    -fy*y/z²]
```

### Implementation
```python
def project_gaussian_3d_to_2d(mean_3d, cov_3d, K, R, t):
    """
    Project a 3D Gaussian to 2D.
    
    Args:
        mean_3d: (3,) 3D mean in world coordinates
        cov_3d: (3, 3) 3D covariance matrix
        K: (3, 3) intrinsic matrix
        R: (3, 3) rotation matrix
        t: (3,) translation vector
    
    Returns:
        mean_2d: (2,) 2D mean in pixel coordinates
        cov_2d: (2, 2) 2D covariance matrix
    """
    # Transform mean to camera coordinates
    mean_cam = R @ mean_3d + t
    x, y, z = mean_cam
    
    # Get intrinsic parameters
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    # Project mean to 2D
    u = fx * x / z + cx
    v = fy * y / z + cy
    mean_2d = np.array([u, v])
    
    # Compute Jacobian of projection
    J = np.array([
        [fx / z,      0,  -fx * x / (z * z)],
        [     0, fy / z,  -fy * y / (z * z)]
    ])
    
    # Transform covariance to camera frame
    cov_cam = R @ cov_3d @ R.T
    
    # Project covariance: Σ' = J * Σ_cam * J^T
    cov_2d = J @ cov_cam @ J.T
    
    return mean_2d, cov_2d
```

### Batch Covariance Projection
```python
def project_gaussians_batch(means_3d, covs_3d, K, R, t):
    """
    Project multiple 3D Gaussians to 2D efficiently.
    
    Args:
        means_3d: (N, 3) array of 3D means
        covs_3d: (N, 3, 3) array of 3D covariance matrices
        K: (3, 3) intrinsic matrix
        R: (3, 3) rotation matrix
        t: (3,) translation vector
    
    Returns:
        means_2d: (N, 2) array of 2D means
        covs_2d: (N, 2, 2) array of 2D covariances
    """
    N = means_3d.shape[0]
    
    # Transform means to camera coordinates
    means_cam = means_3d @ R.T + t  # (N, 3)
    
    # Get intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Project means
    z = means_cam[:, 2]
    means_2d = np.stack([
        fx * means_cam[:, 0] / z + cx,
        fy * means_cam[:, 1] / z + cy
    ], axis=1)  # (N, 2)
    
    # Compute Jacobians for all points
    z2 = z * z
    J = np.zeros((N, 2, 3))
    J[:, 0, 0] = fx / z
    J[:, 0, 2] = -fx * means_cam[:, 0] / z2
    J[:, 1, 1] = fy / z
    J[:, 1, 2] = -fy * means_cam[:, 1] / z2
    
    # Transform covariances to camera frame
    # cov_cam = R @ cov_3d @ R.T for each Gaussian
    covs_cam = np.einsum('ij,njk,lk->nil', R, covs_3d, R)  # (N, 3, 3)
    
    # Project covariances: cov_2d = J @ cov_cam @ J.T
    covs_2d = np.einsum('nij,njk,nlk->nil', J, covs_cam, J)  # (N, 2, 2)
    
    return means_2d, covs_2d
```

## Alternative: poses_bounds.npy Format

The NeRF LLFF format stores camera data differently:

```python
poses_bounds = np.load("poses_bounds.npy")  # Shape: (N_images, 17)

for i in range(len(poses_bounds)):
    pose_data = poses_bounds[i]
    
    # First 15 values: 3x5 pose matrix
    pose_matrix = pose_data[:15].reshape(3, 5)
    
    # Columns 0-2: Rotation matrix (camera axes)
    # Column 3: Translation vector
    # Column 4: [height, width, focal_length]
    R_cols = pose_matrix[:, :3]  # Note: stored as column vectors!
    t = pose_matrix[:, 3]
    hwf = pose_matrix[:, 4]
    height, width, focal = hwf
    
    # Last 2 values: depth bounds
    near, far = pose_data[15:]
    
    # Note: This format uses a different coordinate convention
    # May need to transform to match COLMAP convention
```

## Summary

### Quick Reference

| Information | File | Field |
|-------------|------|-------|
| Focal length | `cameras.bin` | `cam.params[0]` (or `[0:2]` for PINHOLE) |
| Principal point | `cameras.bin` | `cam.params[1:3]` (or `[2:4]` for PINHOLE) |
| Image size | `cameras.bin` | `cam.width`, `cam.height` |
| Rotation (quaternion) | `images.bin` | `img.qvec` |
| Rotation (matrix) | `images.bin` | `img.qvec2rotmat()` |
| Translation | `images.bin` | `img.tvec` |

### Projection Pipeline

1. **Load camera data:**
   ```python
   cameras = read_cameras_binary("sparse/0/cameras.bin")
   images = read_images_binary("sparse/0/images.bin")
   ```

2. **For each view to render:**
   ```python
   img = images[img_id]
   cam = cameras[img.camera_id]
   K = construct_K_matrix(cam)
   R = img.qvec2rotmat()
   t = img.tvec
   ```

3. **Project each Gaussian:**
   ```python
   mean_2d, cov_2d = project_gaussian_3d_to_2d(mean_3d, cov_3d, K, R, t)
   ```

4. **Render:**
   - Sort Gaussians by depth
   - Rasterize each 2D Gaussian with alpha blending

## Additional Notes

### Coordinate Systems
- **COLMAP**: Camera looks along +Z axis, X right, Y down
- **OpenGL/NeRF**: Camera looks along -Z axis, X right, Y up
- You may need coordinate transformations depending on your rendering system

### Distortion
- The intrinsic matrix K assumes **no distortion** (pinhole camera)
- COLMAP models include distortion parameters (e.g., `k` in SIMPLE_RADIAL)
- For accurate projection, apply distortion **after** perspective division
- For Gaussian Splatting, distortion is often ignored or pre-corrected

### Depth Bounds
- Check `depth > 0` to ensure point is in front of camera
- Use near/far planes from `poses_bounds.npy` for depth culling

## Example Scripts

See the repository for complete working examples:
- `explore_camera_data.py`: Inspect camera intrinsics and extrinsics
- `project_gaussians_example.py`: Project 3D points to 2D with examples

## References

- [COLMAP Documentation](https://colmap.github.io/)
- [3D Gaussian Splatting Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [NeRF: Neural Radiance Fields](https://www.matthewtancik.com/nerf)

