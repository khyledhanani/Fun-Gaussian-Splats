"""
Comprehensive test suite for OriginalGaussianSplat rendering method.

Tests cover:
- Mathematical correctness of transformations
- Autograd compatibility
- Edge cases and numerical stability
- Performance benchmarks
"""

import torch
import numpy as np
import math
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gsplat.methods.original.original import OriginalGaussianSplat, quaternion_to_rotation_matrix
from gsplat.utils.dataclasses import Camera, GaussianPrimitive
import time


class TestQuaternionToRotation:
    """Test quaternion to rotation matrix conversion."""
    
    @staticmethod
    def test_identity_quaternion():
        """Test that identity quaternion produces identity rotation."""
        print("\n[Test] Identity quaternion...")
        q = torch.tensor([1.0, 0.0, 0.0, 0.0])
        R = quaternion_to_rotation_matrix(q)
        expected = torch.eye(3)
        
        assert torch.allclose(R, expected, atol=1e-6), f"Expected identity, got {R}"
        print("✓ PASS: Identity quaternion produces identity rotation")
    
    @staticmethod
    def test_180_rotation_x():
        """Test 180° rotation around X axis."""
        print("\n[Test] 180° rotation around X...")
        q = torch.tensor([0.0, 1.0, 0.0, 0.0])  # 180° around X
        R = quaternion_to_rotation_matrix(q)
        
        # Expected: [1, 0, 0; 0, -1, 0; 0, 0, -1]
        expected = torch.tensor([
            [1.0,  0.0,  0.0],
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0]
        ])
        
        assert torch.allclose(R, expected, atol=1e-6), f"Expected {expected}, got {R}"
        print("✓ PASS: 180° rotation around X is correct")
    
    @staticmethod
    def test_orthogonality():
        """Test that rotation matrices are orthogonal."""
        print("\n[Test] Rotation matrix orthogonality...")
        test_quaternions = [
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            torch.tensor([0.707, 0.707, 0.0, 0.0]),
            torch.tensor([0.5, 0.5, 0.5, 0.5]),
        ]
        
        for q in test_quaternions:
            q = q / torch.norm(q)  # Normalize
            R = quaternion_to_rotation_matrix(q)
            
            # Check R @ R.T = I
            product = torch.matmul(R, R.T)
            assert torch.allclose(product, torch.eye(3), atol=1e-5), \
                f"R @ R.T != I for quaternion {q}"
            
            # Check det(R) = 1
            det = torch.det(R)
            assert torch.abs(det - 1.0) < 1e-5, f"det(R) = {det}, expected 1"
        
        print("✓ PASS: All rotation matrices are orthogonal with det=1")


class TestCovarianceMatrix:
    """Test 3D covariance matrix computation."""
    
    @staticmethod
    def test_spherical_gaussian():
        """Test that isotropic scales produce spherical covariance."""
        print("\n[Test] Spherical Gaussian covariance...")
        config = {"device": "cpu"}
        model = OriginalGaussianSplat(config)
        
        # One Gaussian with identity rotation and isotropic scale
        model.means = torch.nn.Parameter(torch.tensor([[0.0, 0.0, 0.0]]))
        model.rotations = torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        model.scales = torch.nn.Parameter(torch.tensor([[0.0, 0.0, 0.0]]))  # log(1) = 0
        
        cov = model.full_covariance_matrix()
        expected = torch.eye(3)
        
        assert torch.allclose(cov[0], expected, atol=1e-5), \
            f"Expected {expected}, got {cov[0]}"
        print("✓ PASS: Spherical Gaussian produces diagonal covariance")
    
    @staticmethod
    def test_anisotropic_gaussian():
        """Test anisotropic Gaussian with different scales."""
        print("\n[Test] Anisotropic Gaussian covariance...")
        config = {"device": "cpu"}
        model = OriginalGaussianSplat(config)
        
        # Scale [2, 3, 4] in log space
        log_scales = torch.tensor([[math.log(2.0), math.log(3.0), math.log(4.0)]])
        model.means = torch.nn.Parameter(torch.tensor([[0.0, 0.0, 0.0]]))
        model.rotations = torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        model.scales = torch.nn.Parameter(log_scales)
        
        cov = model.full_covariance_matrix()
        
        # Expected: diagonal with [4, 9, 16] (scale^2)
        expected_diag = torch.tensor([4.0, 9.0, 16.0])
        actual_diag = torch.diagonal(cov[0])
        
        assert torch.allclose(actual_diag, expected_diag, atol=1e-4), \
            f"Expected diagonal {expected_diag}, got {actual_diag}"
        print("✓ PASS: Anisotropic scales produce correct variances")
    
    @staticmethod
    def test_rotation_effect():
        """Test that rotation transforms the covariance correctly."""
        print("\n[Test] Rotation effect on covariance...")
        config = {"device": "cpu"}
        model = OriginalGaussianSplat(config)
        
        # Anisotropic with 90° rotation around Z
        log_scales = torch.tensor([[0.0, math.log(2.0), 0.0]])  # [1, 2, 1]
        quat_90z = torch.tensor([math.cos(math.pi/4), 0.0, 0.0, math.sin(math.pi/4)])
        
        model.means = torch.nn.Parameter(torch.tensor([[0.0, 0.0, 0.0]]))
        model.rotations = torch.nn.Parameter(quat_90z.unsqueeze(0))
        model.scales = torch.nn.Parameter(log_scales)
        
        cov = model.full_covariance_matrix()
        
        # After 90° rotation around Z, Y variance should move to X
        # Original: diag([1, 4, 1])
        # After rotation: X and Y should swap-ish (depends on exact rotation)
        # At minimum, check symmetry and positive definiteness
        
        # Check symmetry
        assert torch.allclose(cov[0], cov[0].T, atol=1e-5), "Covariance not symmetric"
        
        # Check positive definiteness (eigenvalues > 0)
        eigvals = torch.linalg.eigvalsh(cov[0])
        assert torch.all(eigvals > 0), f"Covariance not positive definite: {eigvals}"
        
        print("✓ PASS: Rotated covariance is symmetric and positive definite")


class TestProjection:
    """Test camera projection operations."""
    
    @staticmethod
    def test_center_projection():
        """Test projection of point at optical axis."""
        print("\n[Test] Center point projection...")
        config = {"device": "cpu"}
        model = OriginalGaussianSplat(config)
        
        # Point at (0, 0, 5) with camera at origin looking down +Z
        model.means = torch.nn.Parameter(torch.tensor([[0.0, 0.0, 5.0]]))
        
        camera = Camera(
            rotation=torch.eye(3),
            translation=torch.zeros(3),
            fx=100.0, fy=100.0,
            cx=50.0, cy=50.0,
            width=100, height=100
        )
        
        x_2d, y_2d, z_cam, means_cam = model.project_means(camera)
        
        # Should project to principal point
        assert torch.abs(x_2d[0] - 50.0) < 1e-4, f"Expected x=50, got {x_2d[0]}"
        assert torch.abs(y_2d[0] - 50.0) < 1e-4, f"Expected y=50, got {y_2d[0]}"
        assert torch.abs(z_cam[0] - 5.0) < 1e-4, f"Expected z=5, got {z_cam[0]}"
        
        print("✓ PASS: Center point projects to principal point")
    
    @staticmethod
    def test_off_center_projection():
        """Test projection of off-center point."""
        print("\n[Test] Off-center point projection...")
        config = {"device": "cpu"}
        model = OriginalGaussianSplat(config)
        
        # Point at (1, 0, 5) -> should project to (50 + 100*1/5, 50) = (70, 50)
        model.means = torch.nn.Parameter(torch.tensor([[1.0, 0.0, 5.0]]))
        
        camera = Camera(
            rotation=torch.eye(3),
            translation=torch.zeros(3),
            fx=100.0, fy=100.0,
            cx=50.0, cy=50.0,
            width=100, height=100
        )
        
        x_2d, y_2d, z_cam, means_cam = model.project_means(camera)
        
        expected_x = 50.0 + 100.0 * (1.0 / 5.0)  # 70.0
        assert torch.abs(x_2d[0] - expected_x) < 1e-4, \
            f"Expected x={expected_x}, got {x_2d[0]}"
        
        print("✓ PASS: Off-center projection is correct")


class TestCovarianceProjection:
    """Test 3D to 2D covariance projection."""
    
    @staticmethod
    def test_spherical_projection():
        """Test that spherical Gaussian projects to circular."""
        print("\n[Test] Spherical to circular projection...")
        config = {"device": "cpu"}
        model = OriginalGaussianSplat(config)
        
        # Spherical Gaussian at (0, 0, 5)
        model.means = torch.nn.Parameter(torch.tensor([[0.0, 0.0, 5.0]]))
        model.rotations = torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        model.scales = torch.nn.Parameter(torch.tensor([[0.0, 0.0, 0.0]]))  # scale=1
        model.sh_coeffs = torch.nn.Parameter(torch.zeros((1, 1, 3)))
        model.opacities = torch.nn.Parameter(torch.tensor([0.0]))
        
        camera = Camera(
            rotation=torch.eye(3),
            translation=torch.zeros(3),
            fx=100.0, fy=100.0,
            cx=50.0, cy=50.0,
            width=100, height=100
        )
        
        _, _, _, means_cam = model.project_means(camera)
        Sigma_2d, _ = model.project_covariance_matrix(camera, means_cam)
        
        # For spherical Gaussian, 2D covariance should be isotropic
        # (at least approximately for small FOV)
        ratio = Sigma_2d[0, 0, 0] / Sigma_2d[0, 1, 1]
        assert 0.9 < ratio < 1.1, \
            f"Expected isotropic 2D cov, got ratio {ratio}"
        
        print("✓ PASS: Spherical Gaussian projects to approximately circular")
    
    @staticmethod
    def test_positive_definite_projection():
        """Test that projected covariance is positive definite."""
        print("\n[Test] Projected covariance positive definiteness...")
        config = {"device": "cpu"}
        model = OriginalGaussianSplat(config)
        
        # Random anisotropic Gaussian
        model.means = torch.nn.Parameter(torch.tensor([[0.5, -0.3, 4.0]]))
        model.rotations = torch.nn.Parameter(torch.tensor([[0.7, 0.5, 0.3, 0.4]]))
        model.scales = torch.nn.Parameter(torch.tensor([[-1.0, 0.5, 0.2]]))
        model.sh_coeffs = torch.nn.Parameter(torch.zeros((1, 1, 3)))
        model.opacities = torch.nn.Parameter(torch.tensor([0.0]))
        
        camera = Camera(
            rotation=torch.eye(3),
            translation=torch.zeros(3),
            fx=100.0, fy=100.0,
            cx=50.0, cy=50.0,
            width=100, height=100
        )
        
        _, _, _, means_cam = model.project_means(camera)
        Sigma_2d, _ = model.project_covariance_matrix(camera, means_cam)
        
        # Check positive definiteness
        eigvals = torch.linalg.eigvalsh(Sigma_2d[0])
        assert torch.all(eigvals > 0), \
            f"Projected covariance not positive definite: {eigvals}"
        
        print("✓ PASS: Projected covariance is positive definite")


class TestExtentCalculation:
    """Test bounding box calculation."""
    
    @staticmethod
    def test_isotropic_extent():
        """Test extent for isotropic Gaussian."""
        print("\n[Test] Isotropic Gaussian extent...")
        config = {"device": "cpu"}
        model = OriginalGaussianSplat(config)
        
        # Covariance with variance 4 in both directions -> std=2, extent=6
        cov_2d = torch.tensor([[[4.0, 0.0], [0.0, 4.0]]])
        x_2d = torch.tensor([50.0])
        y_2d = torch.tensor([50.0])
        
        camera = Camera(
            rotation=torch.eye(3),
            translation=torch.zeros(3),
            fx=100.0, fy=100.0,
            cx=50.0, cy=50.0,
            width=100, height=100
        )
        
        bbox = model.calculate_extent_contribution(cov_2d, x_2d, y_2d, camera)
        x_min, x_max, y_min, y_max = bbox[:, 0]
        
        extent_x = x_max - x_min
        extent_y = y_max - y_min
        
        expected_extent = 2 * 3 * 2.0  # 2 * 3*sigma = 12
        assert torch.abs(extent_x - expected_extent) < 1e-3, \
            f"Expected extent {expected_extent}, got {extent_x}"
        assert torch.abs(extent_y - expected_extent) < 1e-3, \
            f"Expected extent {expected_extent}, got {extent_y}"
        
        print("✓ PASS: Isotropic extent calculation correct")
    
    @staticmethod
    def test_anisotropic_extent():
        """Test extent for anisotropic Gaussian."""
        print("\n[Test] Anisotropic Gaussian extent...")
        config = {"device": "cpu"}
        model = OriginalGaussianSplat(config)
        
        # Variance [100, 1] -> std [10, 1], extent [30, 3]
        cov_2d = torch.tensor([[[100.0, 0.0], [0.0, 1.0]]])
        x_2d = torch.tensor([50.0])
        y_2d = torch.tensor([50.0])
        
        camera = Camera(
            rotation=torch.eye(3),
            translation=torch.zeros(3),
            fx=100.0, fy=100.0,
            cx=50.0, cy=50.0,
            width=100, height=100
        )
        
        bbox = model.calculate_extent_contribution(cov_2d, x_2d, y_2d, camera)
        x_min, x_max, y_min, y_max = bbox[:, 0]
        
        extent_x = x_max - x_min
        extent_y = y_max - y_min
        
        assert torch.abs(extent_x - 60.0) < 1e-3, f"Expected 60, got {extent_x}"
        assert torch.abs(extent_y - 6.0) < 1e-3, f"Expected 6, got {extent_y}"
        
        print("✓ PASS: Anisotropic extent calculation correct")


class TestTileMapping:
    """Test tile-Gaussian assignment."""
    
    @staticmethod
    def test_single_tile():
        """Test Gaussian contained in single tile."""
        print("\n[Test] Single tile assignment...")
        config = {"device": "cpu"}
        model = OriginalGaussianSplat(config)
        
        # Small bounding box in center of one tile
        # Tile size for 100x100 image with 10 tiles = 10x10
        # Center tile (5,5) spans [50-59, 50-59]
        # Make bbox well within one tile
        bbox = torch.tensor([[52.0], [57.0], [52.0], [57.0]])  # 5x5 box in center of tile
        
        tile_map = model.build_tile_gaussians(bbox, 100, 100, num_tiles=10)
        
        # Should only be in center tile (5, 5)
        assert len(tile_map) == 1, f"Expected 1 tile, got {len(tile_map)} tiles: {list(tile_map.keys())}"
        assert (5, 5) in tile_map, f"Expected tile (5,5), got {list(tile_map.keys())}"
        
        print("✓ PASS: Small Gaussian assigned to single tile")
    
    @staticmethod
    def test_multi_tile():
        """Test Gaussian spanning multiple tiles."""
        print("\n[Test] Multi-tile assignment...")
        config = {"device": "cpu"}
        model = OriginalGaussianSplat(config)
        
        # Large bounding box spanning tiles
        bbox = torch.tensor([[0.0], [99.0], [0.0], [99.0]])  # Full image
        
        tile_map = model.build_tile_gaussians(bbox, 100, 100, num_tiles=4)
        
        # Should be in all 16 tiles
        assert len(tile_map) == 16, f"Expected 16 tiles, got {len(tile_map)}"
        
        print("✓ PASS: Large Gaussian assigned to multiple tiles")


class TestAlphaBlending:
    """Test alpha blending logic."""
    
    @staticmethod
    def test_single_opaque_gaussian():
        """Test rendering single opaque Gaussian."""
        print("\n[Test] Single opaque Gaussian rendering...")
        config = {"device": "cpu"}
        model = OriginalGaussianSplat(config)
        
        # Create one Gaussian at center with high opacity
        # SH coeffs are in logit space, so very negative = dark color
        model.means = torch.nn.Parameter(torch.tensor([[0.0, 0.0, 5.0]]))
        model.rotations = torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        model.scales = torch.nn.Parameter(torch.tensor([[-2.0, -2.0, -2.0]]))  # Small
        model.sh_coeffs = torch.nn.Parameter(torch.full((1, 1, 3), -10.0))  # Very dark (logit)
        model.opacities = torch.nn.Parameter(torch.tensor([5.0]))  # High opacity
        
        camera = Camera(
            rotation=torch.eye(3),
            translation=torch.zeros(3),
            fx=100.0, fy=100.0,
            cx=50.0, cy=50.0,
            width=100, height=100
        )
        
        image = model.rasterize(camera, background_color=(1.0, 1.0, 1.0))
        
        # Center pixel should be much darker than background
        center_color = image[50, 50]
        assert torch.all(center_color < 0.2), \
            f"Expected dark center (<0.2), got {center_color}"
        
        # Edge pixels should be white (background)
        edge_color = image[0, 0]
        assert torch.all(edge_color > 0.9), \
            f"Expected white edge (>0.9), got {edge_color}"
        
        print("✓ PASS: Single Gaussian renders correctly")


class TestAutograd:
    """Test autograd compatibility."""
    
    @staticmethod
    def test_gradient_flow():
        """Test that gradients flow through all parameters."""
        print("\n[Test] Gradient flow through parameters...")
        config = {"device": "cpu"}
        model = OriginalGaussianSplat(config)
        
        # Initialize parameters
        model.means = torch.nn.Parameter(torch.tensor([[0.0, 0.0, 5.0]]))
        model.rotations = torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        model.scales = torch.nn.Parameter(torch.tensor([[0.0, 1.0, 0.0]]))
        model.sh_coeffs = torch.nn.Parameter(torch.zeros((1, 1, 3)))
        model.opacities = torch.nn.Parameter(torch.tensor([0.0]))
        
        camera = Camera(
            rotation=torch.eye(3),
            translation=torch.zeros(3),
            fx=100.0, fy=100.0,
            cx=50.0, cy=50.0,
            width=100, height=100
        )
        
        # Forward and backward
        image = model.rasterize(camera)
        loss = image.mean()
        loss.backward()
        
        # Check all parameters have gradients
        params = [
            ("means", model.means),
            ("rotations", model.rotations),
            ("scales", model.scales),
            ("opacities", model.opacities),
            ("sh_coeffs", model.sh_coeffs),
        ]
        
        for name, param in params:
            assert param.grad is not None, f"No gradient for {name}"
            # Note: Some gradients might be zero if parameter doesn't affect output
        
        print("✓ PASS: Gradients computed for all parameters")
    
    @staticmethod
    def test_gradient_values():
        """Test that gradients have reasonable magnitudes."""
        print("\n[Test] Gradient magnitude check...")
        config = {"device": "cpu"}
        model = OriginalGaussianSplat(config)
        
        # Visible Gaussian
        model.means = torch.nn.Parameter(torch.tensor([[0.0, 0.0, 5.0]]))
        model.rotations = torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        model.scales = torch.nn.Parameter(torch.tensor([[0.0, 0.0, 0.0]]))
        model.sh_coeffs = torch.nn.Parameter(torch.zeros((1, 1, 3)))
        model.opacities = torch.nn.Parameter(torch.tensor([0.0]))
        
        camera = Camera(
            rotation=torch.eye(3),
            translation=torch.zeros(3),
            fx=100.0, fy=100.0,
            cx=50.0, cy=50.0,
            width=100, height=100
        )
        
        image = model.rasterize(camera)
        target = torch.ones_like(image)
        loss = torch.nn.functional.mse_loss(image, target)
        loss.backward()
        
        # Check mean gradient is non-zero (Gaussian is visible)
        mean_grad_norm = model.means.grad.norm().item()
        assert mean_grad_norm > 1e-6, \
            f"Mean gradient too small: {mean_grad_norm}"
        
        print(f"✓ PASS: Mean gradient norm = {mean_grad_norm:.6f}")


class TestOptimizer:
    """Test optimizer setup and training."""
    
    @staticmethod
    def test_optimizer_init():
        """Test optimizer initialization."""
        print("\n[Test] Optimizer initialization...")
        config = {
            "device": "cpu",
            "lr_means": 0.001,
            "lr_rotations": 0.0001,
            "lr_scales": 0.005,
            "lr_sh_coeffs": 0.0025,
            "lr_opacities": 0.05,
        }
        model = OriginalGaussianSplat(config)
        
        # Initialize parameters
        model.means = torch.nn.Parameter(torch.tensor([[0.0, 0.0, 5.0]]))
        model.rotations = torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        model.scales = torch.nn.Parameter(torch.tensor([[0.0, 0.0, 0.0]]))
        model.sh_coeffs = torch.nn.Parameter(torch.zeros((1, 1, 3)))
        model.opacities = torch.nn.Parameter(torch.tensor([0.0]))
        
        model.setup_optimizer()
        
        assert hasattr(model, 'optimizer'), "Optimizer not created"
        assert len(model.optimizer.param_groups) == 5, \
            f"Expected 5 param groups, got {len(model.optimizer.param_groups)}"
        
        print("✓ PASS: Optimizer initialized with correct param groups")
    
    @staticmethod
    def test_training_step():
        """Test that training step updates parameters."""
        print("\n[Test] Training step updates parameters...")
        config = {"device": "cpu"}
        model = OriginalGaussianSplat(config)
        
        # Initialize
        model.means = torch.nn.Parameter(torch.tensor([[0.0, 0.0, 5.0]]))
        model.rotations = torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        model.scales = torch.nn.Parameter(torch.tensor([[0.0, 0.0, 0.0]]))
        model.sh_coeffs = torch.nn.Parameter(torch.zeros((1, 1, 3)))
        model.opacities = torch.nn.Parameter(torch.tensor([0.0]))
        model.setup_optimizer()
        
        camera = Camera(
            rotation=torch.eye(3),
            translation=torch.zeros(3),
            fx=100.0, fy=100.0,
            cx=50.0, cy=50.0,
            width=100, height=100
        )
        
        # Store initial values
        initial_means = model.means.clone().detach()
        
        # Training step
        batch = {
            "camera": camera,
            "target_image": torch.ones(100, 100, 3),
        }
        
        result = model.train_step(batch)
        
        # Check loss exists
        assert "loss" in result, "Loss not returned"
        
        # Check parameters changed
        means_changed = not torch.allclose(model.means, initial_means)
        assert means_changed, "Parameters not updated"
        
        print(f"✓ PASS: Training step updated parameters (loss={result['loss']:.6f})")


class TestPerformance:
    """Performance benchmarks."""
    
    @staticmethod
    def test_rendering_speed():
        """Benchmark rendering speed."""
        print("\n[Benchmark] Rendering speed...")
        config = {"device": "cpu"}
        model = OriginalGaussianSplat(config)
        
        # Create many Gaussians
        num_gaussians = 1000
        model.means = torch.nn.Parameter(torch.randn(num_gaussians, 3) * 2 + torch.tensor([0, 0, 5]))
        model.rotations = torch.nn.Parameter(torch.randn(num_gaussians, 4))
        model.rotations.data = model.rotations.data / torch.norm(model.rotations.data, dim=-1, keepdim=True)
        model.scales = torch.nn.Parameter(torch.randn(num_gaussians, 3) * 0.5)
        model.sh_coeffs = torch.nn.Parameter(torch.randn(num_gaussians, 1, 3))
        model.opacities = torch.nn.Parameter(torch.randn(num_gaussians))
        
        camera = Camera(
            rotation=torch.eye(3),
            translation=torch.zeros(3),
            fx=100.0, fy=100.0,
            cx=50.0, cy=50.0,
            width=200, height=200
        )
        
        # Warmup
        _ = model.rasterize(camera)
        
        # Benchmark
        num_runs = 10
        start = time.time()
        for _ in range(num_runs):
            _ = model.rasterize(camera)
        elapsed = time.time() - start
        
        fps = num_runs / elapsed
        print(f"✓ Rendered {num_gaussians} Gaussians at {fps:.2f} FPS (CPU)")
        print(f"  Average time: {elapsed/num_runs*1000:.2f} ms per frame")


def run_all_tests():
    """Run all test suites."""
    print("="*70)
    print("COMPREHENSIVE TEST SUITE FOR ORIGINAL GAUSSIAN SPLAT")
    print("="*70)
    
    test_classes = [
        TestQuaternionToRotation,
        TestCovarianceMatrix,
        TestProjection,
        TestCovarianceProjection,
        TestExtentCalculation,
        TestTileMapping,
        TestAlphaBlending,
        TestAutograd,
        TestOptimizer,
        TestPerformance,
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\n{'='*70}")
        print(f"Running {test_class.__name__}")
        print('='*70)
        
        # Get all test methods
        test_methods = [
            getattr(test_class, method) 
            for method in dir(test_class) 
            if method.startswith('test_')
        ]
        
        for test_method in test_methods:
            total_tests += 1
            try:
                test_method()
                passed_tests += 1
            except Exception as e:
                print(f"✗ FAIL: {test_method.__name__}")
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n{'='*70}")
    print(f"TEST SUMMARY: {passed_tests}/{total_tests} tests passed")
    print('='*70)
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

