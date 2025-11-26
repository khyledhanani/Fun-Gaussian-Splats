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
@METHOD_REGISTRY.register("original")
class OriginalGaussianSplat(GaussianSplatMethod, torch.nn.Module):
	"""3D Gaussian Splatting method.

	Initializes Gaussian primitives from COLMAP point clouds
	and optimizes them via differentiable rasterization, including adaptive
	density control through splitting, cloning, and pruning.

	Attributes:
		state: Arbitrary container for method state.
		primitives: List of Gaussian primitives before batching.
		cameras: List of camera parameters for the dataset.
		images: Metadata for the input images.
		device: Torch device used for computation.
		means: Learnable 3D Gaussian means.
		rotations: Learnable quaternion rotations per Gaussian.
		scales: Learnable log-scales per Gaussian axis.
		sh_coeffs: Learnable spherical harmonic color coefficients.
		opacities: Learnable logit opacities.
		training: Flag indicating whether the module is in training mode.
		step_count: Number of optimization steps performed.
		accumulated_gradients: Buffer used for gradient statistics.
		scene_extent: Approximate diagonal length of the scene bounding box.
	"""
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
		self.scene_extent: float | None = None

	def load_cameras_from_dataset(self, dataset) -> None:
		"""Populate camera and image metadata from a dataset.

		Args:
			dataset: ``ColmapDataset`` instance providing cameras and images.
		"""
		self.cameras = dataset.get_cameras()
		self.images = dataset.get_images()
		print(f"Loaded {len(self.cameras)} cameras and {len(self.images)} images from dataset")
	
	def create_primitives(self, points3d: Dict[int, Point3D]) -> List[GaussianPrimitive]:
		"""Create Gaussian primitives from COLMAP 3D points.

		Args:
			points3d: Mapping from COLMAP point identifiers to ``Point3D`` objects.

		Returns:
			List of initialized ``GaussianPrimitive`` instances.
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
		"""Convert the primitive list into batched learnable tensors so we can use 
		autograd."""
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

		with torch.no_grad():
			mins, _ = means.min(dim=0)
			maxs, _ = means.max(dim=0)
			self.scene_extent = torch.norm(maxs - mins).item()

	def setup_optimizer(self) -> None:
		"""Init optimizer"""
		params = [{'params': [self.means], 'lr': self.config.get("lr_means", 0.001), 'name': 'means'},
				  {'params': [self.rotations], 'lr': self.config.get("lr_rotations", 0.001), 'name': 'rotations'},
				  {'params': [self.scales], 'lr': self.config.get("lr_scales", 0.001), 'name': 'scales'},
				  {'params': [self.sh_coeffs], 'lr': self.config.get("lr_sh_coeffs", 0.001), 'name': 'sh_coeffs'},
				  {'params': [self.opacities], 'lr': self.config.get("lr_opacities", 0.001), 'name': 'opacities'}]

		self.optimizer = torch.optim.Adam(params, lr=self.config.get("lr_total", 0.001), eps=self.config.get("eps", 1e-15))

	def train_step(self, batch: Dict[str, Any], iter_pct: float = 0.0) -> Dict[str, Any]:
		"""Perform a single optimization step.

		Args:
			batch: Dictionary containing at least a ``\"camera\"`` and a
				``\"target_image\"`` tensor.
			iter_pct: Fractional progress through training in ``[0, 1]`` used
				for scheduling adaptive density control.

		Returns:
			Dictionary with the scalar training ``\"loss\"`` and the render
			output dictionary under the key ``\"render\"``.
		"""
		self.training = True
		self.optimizer.zero_grad()

		camera: Camera = batch["camera"]
		background_color = batch.get("background_color", (0.0, 0.0, 0.0))
		render_output = self.forward_render(camera, background_color=background_color)
		
		target_image = batch.get("target_image")
		target_image = target_image.to(self.device)
		loss = F.mse_loss(render_output["image"], target_image)
		loss.backward()
		
		with torch.no_grad():
			if self.means.grad is not None:
				means_cam = torch.matmul(self.means, camera.rotation.T) + camera.translation
				z_cam = means_cam[:, 2].clamp(min=1e-5)
				J = torch.zeros(self.means.shape[0], 2, 3, device=self.device)
				J[:, 0, 0] = camera.fx / z_cam
				J[:, 0, 2] = -(camera.fx * means_cam[:, 0]) / (z_cam ** 2)
				J[:, 1, 1] = camera.fy / z_cam
				J[:, 1, 2] = -(camera.fy * means_cam[:, 1]) / (z_cam ** 2)
				grad_means_cam = torch.matmul(self.means.grad, camera.rotation.T)
				grad_2d = torch.bmm(J, grad_means_cam.unsqueeze(-1)).squeeze(-1)
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
		"""Render an image for a given camera.

		Args:
			camera: Either a ``Camera`` instance or a batch dictionary containing
				a ``\"camera\"`` key and optional ``\"background_color\"``.

		Returns:
			Dictionary containing the rendered RGB image under ``\"image\"`` and
			auxiliary tensors such as projected means.
		"""
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
		"""Compute a forward rendering pass for a single camera."""
		image, x_2d, y_2d = self.rasterize(camera, background_color)
		return {"image": image, "x_2d": x_2d, "y_2d": y_2d}

	def rasterize(self, camera: Camera, background_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""Rasterize the scene for a camera using a tile-based alpha compositor."""
		height, width = camera.height, camera.width
		image = torch.zeros(height, width, 3, device=self.device)
		image[:, :] = torch.tensor(background_color, device=self.device)
		
		if self.means is None or self.means.numel() == 0:
			return image

		x_2d, y_2d, z_cam, means_cam = self.project_means(camera)

		Sigma_2d, precision_2d = self.project_covariance_matrix(camera, means_cam)
		candidate_box = self.calculate_extent_contribution(Sigma_2d, x_2d, y_2d, camera)
		R_inv = camera.rotation.T
		cam_center = -torch.matmul(R_inv, camera.translation)
		view_dirs = self.means - cam_center
		view_dirs = view_dirs / (torch.norm(view_dirs, dim=-1, keepdim=True) + 1e-8)
		colors = eval_sh_color(self.sh_coeffs, view_dirs)
		opacities = torch.sigmoid(self.opacities)

		tile_gaussians = self.build_tile_gaussians(candidate_box, width, height, num_tiles=16)

		tile_width = max(1, width // 16)
		tile_height = max(1, height // 16)

		for (tile_y, tile_x), g_id in tile_gaussians.items():
			if not g_id:
				continue

			g_tensor = torch.tensor(g_id, device=self.device, dtype=torch.long)
			depths = z_cam[g_tensor]
			order = torch.argsort(depths)
			g_sorted = g_tensor[order]

			x_start = tile_x * tile_width
			x_end = min((tile_x + 1) * tile_width, width)
			y_start = tile_y * tile_height
			y_end = min((tile_y + 1) * tile_height, height)
			
			if x_start >= x_end or y_start >= y_end:
				continue

			ys, xs = torch.meshgrid(
				torch.arange(y_start, y_end, device=self.device, dtype=torch.float32),
				torch.arange(x_start, x_end, device=self.device, dtype=torch.float32),
				indexing="ij"
			)
			pixel_coords = torch.stack([xs, ys], dim=-1)
			pixel_coords_flat = pixel_coords.reshape(-1, 2)
			num_pixels = pixel_coords_flat.shape[0]

			tile_means = torch.stack([x_2d[g_sorted], y_2d[g_sorted]], dim=-1)
			tile_prec = precision_2d[g_sorted]
			tile_opac = opacities[g_sorted]
			tile_cols = colors[g_sorted]

			delta = pixel_coords_flat.unsqueeze(0) - tile_means.unsqueeze(1)
			dx = delta[..., 0]
			dy = delta[..., 1]
			p00 = tile_prec[:, 0, 0].unsqueeze(1)
			p01 = tile_prec[:, 0, 1].unsqueeze(1)
			p11 = tile_prec[:, 1, 1].unsqueeze(1)
			power = -0.5 * (dx**2 * p00 + 2 * dx * dy * p01 + dy**2 * p11)
			alpha = tile_opac.unsqueeze(1) * torch.exp(power)
			vis = torch.cumprod(1 - alpha, dim=0)
			ones = torch.ones(1, num_pixels, device=self.device)
			transmittance = torch.cat([ones, vis[:-1]], dim=0)
			weights = alpha * transmittance
			pixel_colors = (weights.unsqueeze(-1) * tile_cols.unsqueeze(1)).sum(dim=0)
			final_T = vis[-1].unsqueeze(-1)
			bg_color = torch.tensor(background_color, device=self.device)
			pixel_colors = pixel_colors + final_T * bg_color
			image[y_start:y_end, x_start:x_end] = pixel_colors.reshape(y_end-y_start, x_end-x_start, 3)

		return image, x_2d, y_2d

	def adaptive_density_control(self, iteration: int, iter_pct: float) -> None:
		"""Apply adaptive density control via splitting, cloning, and pruning.

		Args:
			iteration: Global optimization step index.
			iter_pct: Fractional training progress in ``[0, 1]`` used to gate
				when density control is active.
		"""
		densify_grad_threshold = 0.0002
		min_opacity = 0.005
		if self.scene_extent is None:
			with torch.no_grad():
				mins, _ = self.means.detach().min(dim=0)
				maxs, _ = self.means.detach().max(dim=0)
				self.scene_extent = torch.norm(maxs - mins).item()
		extent = self.scene_extent
		
		# heuristic I used for warm up and cool down
		if (iter_pct < 0.3 or iter_pct > 0.9) or (iter_pct % 0.1 != 0):
			return

		counts = self.grad_2d_count.clamp(min=1)
		avg_grads = self.grad_2d_accumulator / counts
		high_grads = avg_grads > densify_grad_threshold
		scales = torch.exp(self.scales)
		max_scales = torch.max(scales, dim=1).values
		split_mask = high_grads & (max_scales > 0.01 * extent)
		clone_mask = high_grads & (max_scales <= 0.01 * extent)
		new_means = []
		new_rotations = []
		new_scales = []
		new_sh_coeffs = []
		new_opacities = []
		
		if clone_mask.any():
			new_means.append(self.means[clone_mask])
			new_rotations.append(self.rotations[clone_mask])
			new_scales.append(self.scales[clone_mask])
			new_sh_coeffs.append(self.sh_coeffs[clone_mask])
			new_opacities.append(self.opacities[clone_mask])
			
		if split_mask.any():
			n_split = split_mask.sum().item()
			stds = torch.exp(self.scales[split_mask])
			means = self.means[split_mask]
			samples = torch.randn(n_split * 2, 3, device=self.device)
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
			
		prune_mask = (torch.sigmoid(self.opacities) < min_opacity)
		prune_mask = prune_mask | split_mask
		keep_mask = ~prune_mask

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

		self.means = torch.nn.Parameter(final_means)
		self.rotations = torch.nn.Parameter(final_rotations)
		self.scales = torch.nn.Parameter(final_scales)
		self.sh_coeffs = torch.nn.Parameter(final_sh_coeffs)
		self.opacities = torch.nn.Parameter(final_opacities)
		
		self.grad_2d_accumulator = torch.zeros(self.means.shape[0], device=self.device)
		self.grad_2d_count = torch.zeros(self.means.shape[0], device=self.device)
		
		self.setup_optimizer()
		print(f"Step {iteration}: Gaussians: {len(keep_mask)} -> {self.means.shape[0]} (Split: {split_mask.sum()}, Clone: {clone_mask.sum()}, Pruned: {prune_mask.sum()})")	



	def build_tile_gaussians(
		self,
		candidate_box: torch.Tensor,
		width: int,
		height: int,
		num_tiles: int = 16,
	) -> Dict[Tuple[int, int], List[int]]:
		"""Map tiles to the indices of overlapping Gaussians.

		Args:
			candidate_box: Tensor of shape ``(4, N)`` containing ``[x_min, x_max,
				y_min, y_max]`` for each Gaussian in pixels.
			width: Image width in pixels.
			height: Image height in pixels.
			num_tiles: Number of tiles along each image dimension.

		Returns:
			Dictionary mapping ``(tile_y, tile_x)`` indices to lists of Gaussian
			indices contributing to that tile.
		"""
		x_min_f, x_max_f, y_min_f, y_max_f = candidate_box
		num_gaussians = x_min_f.shape[0]

		tile_width = max(1, width // num_tiles)
		tile_height = max(1, height // num_tiles)

		tile_gaussians: Dict[Tuple[int, int], List[int]] = {}

		for g in range(num_gaussians):
			gx_min = int(torch.floor(x_min_f[g]).item())
			gx_max = int(torch.ceil(x_max_f[g]).item())
			gy_min = int(torch.floor(y_min_f[g]).item())
			gy_max = int(torch.ceil(y_max_f[g]).item())

			gx_min = max(0, min(width - 1, gx_min))
			gx_max = max(0, min(width - 1, gx_max))
			gy_min = max(0, min(height - 1, gy_min))
			gy_max = max(0, min(height - 1, gy_max))

			if gx_max < gx_min or gy_max < gy_min:
				continue

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
		"""Project 3D Gaussian means to 2D image coordinates.

		Args:
			camera: Camera parameters used for projection.

		Returns:
			Tuple ``(x_2d, y_2d, z_cam, means_cam)`` where ``x_2d`` and ``y_2d``
			are pixel coordinates, ``z_cam`` is depth in camera space, and
			``means_cam`` are means expressed in camera coordinates.
		"""
		means_cam = torch.matmul(self.means, camera.rotation.T) + camera.translation

		z_cam = means_cam[:, 2]
		x_2d = (means_cam[:, 0] / z_cam) * camera.fx + camera.cx
		y_2d = (means_cam[:, 1] / z_cam) * camera.fy + camera.cy

		return x_2d, y_2d, z_cam, means_cam
	
	def full_covariance_matrix(self):
		"""Compute full 3D covariance matrices for all Gaussians."""
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
		"""Project 3D covariance matrices into 2D image space.

		Args:
			camera: Camera providing intrinsics and extrinsics.
			means_cam: Gaussian means in camera coordinates, shape ``(N, 3)``.

		Returns:
			Tuple ``(Sigma_2d, precision_2d)`` where both tensors have shape
			``(N, 2, 2)`` and represent the 2D covariance and its inverse.
		"""
		Sigma_world = self.full_covariance_matrix()
		R_cam = camera.rotation.to(means_cam.device) 
		Sigma_cam = torch.matmul(R_cam, torch.matmul(Sigma_world, R_cam.T))
		x, y, z = means_cam[:, 0], means_cam[:, 1], means_cam[:, 2]
		fx = camera.fx
		fy = camera.fy
		z = torch.clamp(z, min=1e-5)
		z2 = z * z
		J = torch.zeros(means_cam.shape[0], 2, 3, device=self.device)
		J[:, 0, 0] = fx / z
		J[:, 0, 2] = -(fx * x) / z2
		J[:, 1, 1] = fy / z
		J[:, 1, 2] = -(fy * y) / z2
		
		Sigma_2d = torch.bmm(J, torch.bmm(Sigma_cam, J.transpose(1, 2)))

		precision_2d = torch.linalg.inv(Sigma_2d + 1e-8 * torch.eye(2, dtype=Sigma_2d.dtype, device=Sigma_2d.device))
		
		return Sigma_2d, precision_2d

	def calculate_extent_contribution(self, cov_2d: torch.Tensor, x_2d: torch.Tensor, y_2d: torch.Tensor, camera: Camera) -> torch.Tensor:
		"""Compute conservative 2D support boxes for each Gaussian.

		Args:
			cov_2d: 2D covariance matrices of shape ``(N, 2, 2)``.
			x_2d: Projected x-coordinates in pixels.
			y_2d: Projected y-coordinates in pixels.
			camera: Camera providing image dimensions.

		Returns:
			Tensor of shape ``(4, N)`` containing ``[x_min, x_max, y_min, y_max]``
			for each Gaussian in pixel coordinates.
		"""
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




