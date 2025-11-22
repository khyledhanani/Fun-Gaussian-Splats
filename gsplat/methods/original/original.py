from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
from tqdm.auto import tqdm

import torch
import numpy as np
import torch.nn.functional as F
from gsplat.datasets import Point3D, read_points3D_binary
from gsplat.methods.base import GaussianSplatMethod
from gsplat.utils.dataclasses import GaussianPrimitive, eval_sh_color
from gsplat.registry import METHOD_REGISTRY
from gsplat.utils.dataclasses import Camera, Image


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
	"""
	Convert quaternion (w, x, y, z) to 3x3 rotation matrix.
	
	Args:
		q: Quaternion tensor of shape (4,) with [w, x, y, z]
	
	Returns:
		3x3 rotation matrix
	"""
	w, x, y, z = q[0], q[1], q[2], q[3]
	
	# Build rotation matrix (quaternion is already normalized in __post_init__)
	R = torch.stack([
		torch.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)]),
		torch.stack([2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)]),
		torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)])
	])
	
	return R


@METHOD_REGISTRY.register("original")
class OriginalGaussianSplat(GaussianSplatMethod, torch.nn.Module):
	def __init__(self, config: Dict[str, Any]) -> None:
		GaussianSplatMethod.__init__(self, config)
		torch.nn.Module.__init__(self)
		print(f"Initializing OriginalGaussianSplat with config: {self.config}")
		self.state: Dict[str, Any] = {}
		self.primitives: List[GaussianPrimitive] = []
		self.cameras: List[Camera] = []
		self.images: List[Image] = []
		self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
		self.means = None
		self.rotations = None
		self.scales = None
		self.sh_coeffs = None
		self.opacities = None
		self.training = False
		self.step_count = 0
		self.accumulated_gradients: Dict[str, torch.Tensor] = {}

	def load_cameras_from_dataset(self, dataset) -> None:
		"""
		Load cameras and images from a ColmapDataset.
		
		Args:
			dataset: ColmapDataset instance with loaded camera data
		"""
		self.cameras = dataset.get_cameras()
		self.images = dataset.get_images()
		print(f"Loaded {len(self.cameras)} cameras and {len(self.images)} images from dataset")
	
	def create_primitives(self, points3d: Dict[int, Point3D]) -> List[GaussianPrimitive]:
		"""
		Create Gaussian primitives from COLMAP 3D points.
		
		Args:
			points3d: Dictionary mapping point IDs to Point3D objects from COLMAP
		
		Returns:
			List of GaussianPrimitive objects initialized from the point cloud
		"""
		print(f"Creating {len(points3d)} primitives from COLMAP 3D points")
		primitives_list: List[GaussianPrimitive] = []
		gaussian_config = self.config.get("gaussian", {})
		initial_scale = gaussian_config.get("initial_scale", 0.01)
		sh_degree = gaussian_config.get("sh_degree", 0)  

		for point3d in tqdm(points3d.values(), desc="Creating primitives"):
			primitive = GaussianPrimitive.from_point3d(
				position=point3d.xyz,
				color=point3d.rgb,
				initial_scale=initial_scale,
				sh_degree=sh_degree,
				device=self.device,
			)
			primitives_list.append(primitive)
		self.primitives = primitives_list

		self.convert_primitives_to_batches()

	def convert_primitives_to_batches(self) -> List[Dict[str, Any]]:
		"""
		"""
		means = torch.stack([primitive.position for primitive in self.primitives])
		rotations = torch.stack([primitive.rotation for primitive in self.primitives])
		scales = torch.stack([primitive.scale for primitive in self.primitives])
		sh_coeffs = torch.stack([primitive.sh_coeffs for primitive in self.primitives])
		opacities = torch.stack([primitive.opacity for primitive in self.primitives])

		self.means = torch.nn.Parameter(means)
		self.rotations = torch.nn.Parameter(rotations)
		self.scales = torch.nn.Parameter(scales)
		self.sh_coeffs = torch.nn.Parameter(sh_coeffs)
		self.opacities = torch.nn.Parameter(opacities)

		self.grad_2d_accumulator = torch.zeros(means.shape[0], device=self.device)
		self.grad_2d_count = torch.zeros(means.shape[0], device=self.device)

	def setup_optimizer(self) -> None:
		print("Setting up optimizer")
		params = [{'params': [self.means], 'lr': self.config.get("lr_means", 0.001), 'name': 'means'},
				  {'params': [self.rotations], 'lr': self.config.get("lr_rotations", 0.001), 'name': 'rotations'},
				  {'params': [self.scales], 'lr': self.config.get("lr_scales", 0.001), 'name': 'scales'},
				  {'params': [self.sh_coeffs], 'lr': self.config.get("lr_sh_coeffs", 0.001), 'name': 'sh_coeffs'},
				  {'params': [self.opacities], 'lr': self.config.get("lr_opacities", 0.001), 'name': 'opacities'}]

		self.optimizer = torch.optim.Adam(params, lr=self.config.get("lr_total", 0.001), eps=self.config.get("eps", 1e-15))

	def train_step(self, batch: Dict[str, Any], iter_pct: float = 0.0) -> Dict[str, Any]:
		self.training = True
		self.optimizer.zero_grad()

		camera: Camera = batch["camera"]
		background_color = batch.get("background_color", (0.0, 0.0, 0.0))
		render_output = self.forward_render(camera, background_color=background_color)
		
		# loss computation
		target_image = batch.get("target_image")
		target_image = target_image.to(self.device)
		loss = F.mse_loss(render_output["image"], target_image)
		loss.backward()
		
		# Accumulate 2D positional gradients for densification
		# Compute viewspace gradient by projecting 3D mean gradients to 2D
		with torch.no_grad():
			if self.means.grad is not None:
				# Get camera parameters for projection
				means_cam = torch.matmul(self.means, camera.rotation.T) + camera.translation
				z_cam = means_cam[:, 2].clamp(min=1e-5)
				
				# Jacobian of 2D projection w.r.t. camera-space means
				# dx_2d/dmeans_cam and dy_2d/dmeans_cam
				J = torch.zeros(self.means.shape[0], 2, 3, device=self.device)
				J[:, 0, 0] = camera.fx / z_cam
				J[:, 0, 2] = -(camera.fx * means_cam[:, 0]) / (z_cam ** 2)
				J[:, 1, 1] = camera.fy / z_cam
				J[:, 1, 2] = -(camera.fy * means_cam[:, 1]) / (z_cam ** 2)
				
				# Transform 3D gradient to camera space
				grad_means_cam = torch.matmul(self.means.grad, camera.rotation.T)
				
				# Project to 2D: grad_2d = J @ grad_means_cam
				grad_2d = torch.bmm(J, grad_means_cam.unsqueeze(-1)).squeeze(-1)  # (N, 2)
				grad_2d_norm = torch.norm(grad_2d, dim=1)
				
				self.grad_2d_accumulator += grad_2d_norm
				self.grad_2d_count += 1
		
		self.optimizer.step()
		self.step_count += 1
		self.adaptive_density_control(self.step_count, iter_pct)

		return {
			"loss": loss.item(),
			"render": render_output,
		}
		

	def render(self, camera: Dict[str, Any] | Camera) -> Dict[str, Any]:
		if isinstance(camera, Camera):
			cam_obj = camera
			background_color = self.config.get("background_color", (0.0, 0.0, 0.0))
		else:
			cam_obj = camera["camera"]
			background_color = camera.get(
				"background_color", self.config.get("background_color", (0.0, 0.0, 0.0))
			)

		return self.forward_render(cam_obj, background_color=background_color)
	

	def forward_render(self, camera: Camera, background_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Dict[str, Any]:
		"""
		Forward pass for rendering.
		"""
		image, x_2d, y_2d = self.rasterize(camera, background_color)
		return {"image": image, "x_2d": x_2d, "y_2d": y_2d}

	def rasterize(self, camera: Camera, background_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		Rasterize the scene using a simple tile-based scheme.
		"""
		height, width = camera.height, camera.width
		image = torch.zeros(height, width, 3, device=self.device)
		# Start with background color
		image[:, :] = torch.tensor(background_color, device=self.device)
		
		if self.means is None or self.means.numel() == 0:
			return image

		x_2d, y_2d, z_cam, means_cam = self.project_means(camera)

		# 3D -> 2D covariance and candidate boxes
		Sigma_2d, precision_2d = self.project_covariance_matrix(camera, means_cam)
		candidate_box = self.calculate_extent_contribution(Sigma_2d, x_2d, y_2d, camera)
		
		# Pre-compute colors for all Gaussians
		# Camera center in world space: C = -R^T @ t
		R_inv = camera.rotation.T
		cam_center = -torch.matmul(R_inv, camera.translation)
		
		# View direction from Gaussian to Camera (or Camera to Gaussian?)
		# SH usually assumes view direction is vector FROM point TO camera (for view dependent effects)
		# If we use FROM camera TO point, we might need to negate.
		# GaussianPrimitive.get_color implementation just takes a direction.
		# Standard 3DGS uses dir = means - cam_center
		view_dirs = self.means - cam_center
		view_dirs = view_dirs / (torch.norm(view_dirs, dim=-1, keepdim=True) + 1e-8)
		
		# Use the shared SH evaluation function
		colors = eval_sh_color(self.sh_coeffs, view_dirs)
		
		# Opacities (sigmoid applied)
		opacities = torch.sigmoid(self.opacities)

		# --- 2. Build tile -> gaussian index mapping based on box overlap ---
		# num_tiles=16 means 16x16 tiles? Or 16 along one dimension? 
		# build_tile_gaussians uses num_tiles along each axis.
		tile_gaussians = self.build_tile_gaussians(candidate_box, width, height, num_tiles=16)

		# Tile size
		tile_width = max(1, width // 16)
		tile_height = max(1, height // 16)

		for (tile_y, tile_x), g_id in tile_gaussians.items():
			if not g_id:
				continue
				
			# Sort by depth
			g_tensor = torch.tensor(g_id, device=self.device, dtype=torch.long)
			depths = z_cam[g_tensor]
			order = torch.argsort(depths) # front-to-back? z_cam is positive in front of camera? 
			# In standard convention (Z+ forward), smaller Z is closer.
			# If Z+ is depth, we want small to large (Painter's algorithm is back-to-front, 
			# but Alpha Blending is front-to-back usually for efficiency or back-to-front).
			# Standard 3DGS is front-to-back alpha blending.
			g_sorted = g_tensor[order]

			# Get pixel coordinates for this tile
			# Clip to image bounds
			x_start = tile_x * tile_width
			x_end = min((tile_x + 1) * tile_width, width)
			y_start = tile_y * tile_height
			y_end = min((tile_y + 1) * tile_height, height)
			
			if x_start >= x_end or y_start >= y_end:
				continue
				
			# Create meshgrid of pixels
			ys, xs = torch.meshgrid(
				torch.arange(y_start, y_end, device=self.device, dtype=torch.float32),
				torch.arange(x_start, x_end, device=self.device, dtype=torch.float32),
				indexing="ij"
			)
			# xs, ys are (H_tile, W_tile)
			pixel_coords = torch.stack([xs, ys], dim=-1) # (H_t, W_t, 2)
			pixel_coords_flat = pixel_coords.reshape(-1, 2) # (P, 2)
			num_pixels = pixel_coords_flat.shape[0]

			# Gather Gaussian parameters
			# Means: (G, 2)
			tile_means = torch.stack([x_2d[g_sorted], y_2d[g_sorted]], dim=-1)
			# Precision: (G, 2, 2)
			tile_prec = precision_2d[g_sorted]
			# Opacity: (G,)
			tile_opac = opacities[g_sorted]
			# Color: (G, 3)
			tile_cols = colors[g_sorted]

			# Vectorized Alpha computation: (G, P)
			# delta = pixel - mean
			# We want (G, P, 2)
			# pixel_coords_flat: (1, P, 2)
			# tile_means: (G, 1, 2)
			delta = pixel_coords_flat.unsqueeze(0) - tile_means.unsqueeze(1) # (G, P, 2)
			
			# Power = -0.5 * delta.T @ Sigma_inv @ delta
			# Efficient computation:
			# Sigma_inv is symmetric? Yes.
			# val = dx^2 * P00 + 2*dx*dy*P01 + dy^2 * P11
			dx = delta[..., 0]
			dy = delta[..., 1]
			p00 = tile_prec[:, 0, 0].unsqueeze(1) # (G, 1)
			p01 = tile_prec[:, 0, 1].unsqueeze(1)
			p11 = tile_prec[:, 1, 1].unsqueeze(1)
			
			power = -0.5 * (dx**2 * p00 + 2 * dx * dy * p01 + dy**2 * p11)
			
			# Filter out small contributions (optional optimization)
			# But for autograd, masking can be tricky if not careful.
			# Let's keep it dense for now.
			
			alpha = tile_opac.unsqueeze(1) * torch.exp(power) # (G, P)
			
			# Accumulate
			# T_i = prod_{j<i} (1 - alpha_j)
			# This is hard to vectorize purely across G without a loop or cumprod
			# Use cumprod
			
			# Clamp alpha to [0, 1) to avoid log(0) issues if we worked in log space, 
			# but here we use (1 - alpha). alpha can be > 1 theoretically if opacity > 1 or precision large? 
			# Sigmoid caps opacity at 1. Exp(power) <= 1 since power <= 0 (precision is positive definite).
			# So alpha <= 1.
			
			# Transmittance calculation
			# T_0 = 1
			# T_i = T_{i-1} * (1 - alpha_{i-1})
			# T = cumprod(1 - alpha) shifted
			
			vis = torch.cumprod(1 - alpha, dim=0) # (G, P)
			# T_i corresponds to vis[i-1]
			# Prepend 1.0 row
			ones = torch.ones(1, num_pixels, device=self.device)
			transmittance = torch.cat([ones, vis[:-1]], dim=0) # (G, P)
			
			weights = alpha * transmittance # (G, P)
			
			# Sum for final color
			# weights: (G, P) -> (G, P, 1)
			# tile_cols: (G, 3) -> (G, 1, 3)
			pixel_colors = (weights.unsqueeze(-1) * tile_cols.unsqueeze(1)).sum(dim=0) # (P, 3)
			
			# Background blending
			# Final T for background
			final_T = vis[-1].unsqueeze(-1) # (P, 1)
			bg_color = torch.tensor(background_color, device=self.device)
			pixel_colors = pixel_colors + final_T * bg_color
			
			# Place back in image
			image[y_start:y_end, x_start:x_end] = pixel_colors.reshape(y_end-y_start, x_end-x_start, 3)

		return image, x_2d, y_2d

	def adaptive_density_control(self, iteration: int, iter_pct: float) -> None:
		"""
		Implements densification and pruning.
		"""
		# Hyperparameters (should be in config ideally)
		densify_grad_threshold = 0.0002
		min_opacity = 0.005
		extent = 5.0 # Approximate scene extent, ideally computed from initial points
		
		# Only densify after warm-up and periodically
		if iter_pct < 0.3 or iter_pct > 0.8:
			return

		# 1. Compute average gradient magnitude
		# Avoid division by zero
		counts = self.grad_2d_count.clamp(min=1)
		avg_grads = self.grad_2d_accumulator / counts
		
		# 2. Identify candidates for densification
		high_grads = avg_grads > densify_grad_threshold
		
		# Split vs Clone based on scale
		# Extract max scale for each Gaussian
		scales = torch.exp(self.scales)
		max_scales = torch.max(scales, dim=1).values
		
		# Heuristic: Split if scale is too large (e.g. > 1% of extent)
		split_mask = high_grads & (max_scales > 0.01 * extent)
		clone_mask = high_grads & (max_scales <= 0.01 * extent)
		
		# 3. Densify
		new_means = []
		new_rotations = []
		new_scales = []
		new_sh_coeffs = []
		new_opacities = []
		
		# Clone
		if clone_mask.any():
			new_means.append(self.means[clone_mask])
			new_rotations.append(self.rotations[clone_mask])
			new_scales.append(self.scales[clone_mask])
			new_sh_coeffs.append(self.sh_coeffs[clone_mask])
			new_opacities.append(self.opacities[clone_mask])
			
		# Split
		if split_mask.any():
			# Create 2 new gaussians for each split one
			# Sample positions from the Gaussian distribution
			n_split = split_mask.sum().item()
			stds = torch.exp(self.scales[split_mask])
			means = self.means[split_mask]
			
			# Sample 2 points
			samples = torch.randn(n_split * 2, 3, device=self.device)
			# We need to rotate these samples according to the rotation of the gaussians
			# This is a simplification; rigorous sampling would rotate the std-aligned samples
			# But typically we just reduce scale and duplicate means with slight offset
			
			# Simple Split: Copy twice, reduce scale by factor of 1.6 (log scale - log(1.6))
			split_means = means.repeat(2, 1)
			split_rotations = self.rotations[split_mask].repeat(2, 1)
			split_scales = self.scales[split_mask].repeat(2, 1) - np.log(1.6)
			split_sh = self.sh_coeffs[split_mask].repeat(2, 1, 1)
			split_opac = self.opacities[split_mask].repeat(2)
			
			new_means.append(split_means)
			new_rotations.append(split_rotations)
			new_scales.append(split_scales)
			new_sh_coeffs.append(split_sh)
			new_opacities.append(split_opac)
			
			# Mark original split ones for removal (optional, or just keep them?)
			# Standard 3DGS replaces them. Here we are appending new ones.
			# If we append, we should remove the old 'split' ones.
			# Let's handle removal in the pruning step or by rebuilding the whole list.
			
		# 4. Prune
		# Remove opacity too low
		prune_mask = (torch.sigmoid(self.opacities) < min_opacity)
		
		# Also prune the ones we just split? 
		# The paper says "split ... and replace".
		prune_mask = prune_mask | split_mask
		
		# Keep mask
		keep_mask = ~prune_mask
		
		# Apply changes
		if len(new_means) > 0:
			added_means = torch.cat(new_means)
			added_rotations = torch.cat(new_rotations)
			added_scales = torch.cat(new_scales)
			added_sh_coeffs = torch.cat(new_sh_coeffs)
			added_opacities = torch.cat(new_opacities)
			
			final_means = torch.cat([self.means[keep_mask], added_means])
			final_rotations = torch.cat([self.rotations[keep_mask], added_rotations])
			final_scales = torch.cat([self.scales[keep_mask], added_scales])
			final_sh_coeffs = torch.cat([self.sh_coeffs[keep_mask], added_sh_coeffs])
			final_opacities = torch.cat([self.opacities[keep_mask], added_opacities])
		else:
			final_means = self.means[keep_mask]
			final_rotations = self.rotations[keep_mask]
			final_scales = self.scales[keep_mask]
			final_sh_coeffs = self.sh_coeffs[keep_mask]
			final_opacities = self.opacities[keep_mask]
			
		# Update parameters
		self.means = torch.nn.Parameter(final_means)
		self.rotations = torch.nn.Parameter(final_rotations)
		self.scales = torch.nn.Parameter(final_scales)
		self.sh_coeffs = torch.nn.Parameter(final_sh_coeffs)
		self.opacities = torch.nn.Parameter(final_opacities)
		
		# Reset stats
		self.grad_2d_accumulator = torch.zeros(self.means.shape[0], device=self.device)
		self.grad_2d_count = torch.zeros(self.means.shape[0], device=self.device)
		
		# Re-init optimizer with new parameters
		self.setup_optimizer()
		print(f"Step {iteration}: Gaussians: {len(keep_mask)} -> {self.means.shape[0]} (Split: {split_mask.sum()}, Clone: {clone_mask.sum()}, Pruned: {prune_mask.sum()})")	



	def build_tile_gaussians(
		self,
		candidate_box: torch.Tensor,
		width: int,
		height: int,
		num_tiles: int = 16,
	) -> Dict[Tuple[int, int], List[int]]:
		"""
		Build a mapping from (tile_y, tile_x) to the list of gaussian indices
		whose candidate boxes overlap that tile.

		Args:
			candidate_box: Tensor of shape (4, N) with [x_min, x_max, y_min, y_max]
			width: Image width in pixels
			height: Image height in pixels
			num_tiles: Number of tiles along each axis (image is split into num_tiles x num_tiles)
		"""
		# candidate_box: (4, N) -> x_min, x_max, y_min, y_max
		x_min_f, x_max_f, y_min_f, y_max_f = candidate_box
		num_gaussians = x_min_f.shape[0]

		tile_width = max(1, width // num_tiles)
		tile_height = max(1, height // num_tiles)

		tile_gaussians: Dict[Tuple[int, int], List[int]] = {}

		for g in range(num_gaussians):
			# Convert floating candidate box to integer pixel bounds and clamp to image
			gx_min = int(torch.floor(x_min_f[g]).item())
			gx_max = int(torch.ceil(x_max_f[g]).item())
			gy_min = int(torch.floor(y_min_f[g]).item())
			gy_max = int(torch.ceil(y_max_f[g]).item())

			gx_min = max(0, min(width - 1, gx_min))
			gx_max = max(0, min(width - 1, gx_max))
			gy_min = max(0, min(height - 1, gy_min))
			gy_max = max(0, min(height - 1, gy_max))

			# Skip if box collapsed
			if gx_max < gx_min or gy_max < gy_min:
				continue

			# Determine tiles overlapped by this bounding box
			tile_x0 = gx_min // tile_width
			tile_x1 = gx_max // tile_width
			tile_y0 = gy_min // tile_height
			tile_y1 = gy_max // tile_height

			tile_x0 = max(0, min(num_tiles - 1, tile_x0))
			tile_x1 = max(0, min(num_tiles - 1, tile_x1))
			tile_y0 = max(0, min(num_tiles - 1, tile_y0))
			tile_y1 = max(0, min(num_tiles - 1, tile_y1))

			for tile_y in range(tile_y0, tile_y1 + 1):
				for tile_x in range(tile_x0, tile_x1 + 1):
					key = (tile_y, tile_x)
					if key not in tile_gaussians:
						tile_gaussians[key] = []
					tile_gaussians[key].append(g)

		return tile_gaussians

	def project_means(self, camera: Camera) -> torch.Tensor:
		"""
		Project the means of the primitives to 2D image coordinates using camera intrinsics and extrinsics.
		"""
		# world to camera coordinates
		means_cam = torch.matmul(self.means, camera.rotation.T) + camera.translation

		# pinhole projection
		z_cam = means_cam[:, 2]
		x_2d = (means_cam[:, 0] / z_cam) * camera.fx + camera.cx
		y_2d = (means_cam[:, 1] / z_cam) * camera.fy + camera.cy

		return x_2d, y_2d, z_cam, means_cam
	
	def full_covariance_matrix(self):
		rots = self.rotations / (torch.norm(self.rotations, dim=-1, keepdim=True) + 1e-8)
		w, x, y, z = rots.unbind(-1)
		xx = x * x
		yy = y * y
		zz = z * z
		xy = x * y
		xz = x * z
		yz = y * z
		wx = w * x
		wy = w * y
		wz = w * z
		row0 = torch.stack([1 - 2*(yy + zz),     2*(xy - wz),     2*(xz + wy)], dim=-1)
		row1 = torch.stack([    2*(xy + wz), 1 - 2*(xx + zz),     2*(yz - wx)], dim=-1)
		row2 = torch.stack([    2*(xz - wy),     2*(yz + wx), 1 - 2*(xx + yy)], dim=-1)

		R = torch.stack([row0, row1, row2], dim=1)
		scales = torch.exp(self.scales)
		S_matrices = torch.diag_embed(scales ** 2)

		RS = torch.bmm(R, S_matrices)
		RSR = torch.bmm(RS, R.transpose(1, 2))
		return RSR

	def project_covariance_matrix(self, camera: Camera, means_cam: torch.Tensor) -> torch.Tensor:
		"""
		Compute and project the 3D covariances to 2D image space.
		"""
		# 1. Get World Space Covariance: (N, 3, 3)
		Sigma_world = self.full_covariance_matrix()
		
		# 2. Transform to Camera Space: Sigma_cam = R_cam @ Sigma_world @ R_cam.T
		R_cam = camera.rotation.to(means_cam.device) 
		
		# R_cam is (3, 3), broadcast against (N, 3, 3)
		# We want R_cam @ Sigma_world @ R_cam.T for each matrix in the batch
		Sigma_cam = torch.matmul(R_cam, torch.matmul(Sigma_world, R_cam.T))
		
		# 3. Compute Jacobians for Projection
		x, y, z = means_cam[:, 0], means_cam[:, 1], means_cam[:, 2]
		
		fx = camera.fx
		fy = camera.fy
		
		# Avoid division by zero
		z = torch.clamp(z, min=1e-5)
		z2 = z * z
		
		# Construct the Jacobian matrix J (N, 2, 3)
		J = torch.zeros(means_cam.shape[0], 2, 3, device=self.device)
		
		J[:, 0, 0] = fx / z
		J[:, 0, 2] = -(fx * x) / z2
		J[:, 1, 1] = fy / z
		J[:, 1, 2] = -(fy * y) / z2
		
		# 4. Project: Sigma_2d = J @ Sigma_cam @ J.T
		Sigma_2d = torch.bmm(J, torch.bmm(Sigma_cam, J.transpose(1, 2)))

		precision_2d = torch.linalg.inv(Sigma_2d + 1e-8 * torch.eye(2, dtype=Sigma_2d.dtype, device=Sigma_2d.device))
		
		return Sigma_2d, precision_2d

	def calculate_extent_contribution(self, cov_2d: torch.Tensor, x_2d: torch.Tensor, y_2d: torch.Tensor, camera: Camera) -> torch.Tensor:
		"""
		Calculate the extent of the 2D covariance matrix.
		"""
		# Use 3-sigma bound based on marginal variances
		extent_x = 3 * torch.sqrt(cov_2d[:, 0, 0])
		extent_y = 3 * torch.sqrt(cov_2d[:, 1, 1])

		x_min = torch.clamp(x_2d - extent_x, 0, camera.width - 1)
		x_max = torch.clamp(x_2d + extent_x, 0, camera.width - 1)
		y_min = torch.clamp(y_2d - extent_y, 0, camera.height - 1)
		y_max = torch.clamp(y_2d + extent_y, 0, camera.height - 1)
		candidate_box = torch.stack([x_min, x_max, y_min, y_max], dim=0)

		return candidate_box


	def save(self, path: Path) -> None:
		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_bytes(b"placeholder")

	def load(self, path: Path, strict: bool = True) -> None:
		points3d = read_points3D_binary(path)




