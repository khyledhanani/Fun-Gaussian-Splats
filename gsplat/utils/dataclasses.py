"""Gaussian primitive data structure for 3D Gaussian Splatting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union

import numpy as np
import torch


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
	"""Convert a quaternion to a 3x3 rotation matrix.

	Args:
		q: Tensor of shape ``(4,)`` containing the quaternion in ``(w, x, y, z)`` order.

	Returns:
		Tensor of shape ``(3, 3)`` representing the corresponding rotation matrix.
	"""
	w, x, y, z = q[0], q[1], q[2], q[3]
	R = torch.stack(
		[
			torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)]),
			torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)]),
			torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]),
		]
	)

	return R


def get_sh_degree(num_coeffs: int) -> int:
	"""Get spherical harmonics degree from number of coefficients per channel."""
	# Number of coeffs for degree n is (n+1)^2
	# So we solve: (n+1)^2 = num_coeffs
	n = int(math.sqrt(num_coeffs)) - 1
	return n


def get_num_sh_coeffs(degree: int) -> int:
	"""Get number of spherical harmonics coefficients for a given degree."""
	return (degree + 1) ** 2


def evaluate_sh_basis(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, degree: int, device: torch.device | None = None) -> torch.Tensor:
	"""
	Evaluate spherical harmonics basis functions up to given degree.
	Works with batched inputs.
	
	Uses the standard SH basis functions:
	- Y_0^0 = 0.282095 (constant)
	- Y_1^{-1} = 0.488603 * y
	- Y_1^0 = 0.488603 * z
	- Y_1^1 = 0.488603 * x
	- etc.
	
	Args:
		x, y, z: Normalized direction vector components (can be scalar or batched tensors)
		degree: Maximum SH degree
		device: Device to create tensors on (inferred from x if None)
	
	Returns:
		Tensor of SH basis values, shape (..., num_coeffs) where ... matches input shape
	"""
	num_coeffs = get_num_sh_coeffs(degree)
	
	if device is None:
		device = x.device if isinstance(x, torch.Tensor) else torch.device('cpu')
	
	# Determine output shape
	if isinstance(x, torch.Tensor):
		output_shape = list(x.shape) + [num_coeffs]
	else:
		output_shape = [num_coeffs]
	
	# Create sh tensor on appropriate device
	sh = torch.zeros(output_shape, dtype=torch.float32, device=device)
	
	# Degree 0
	sh[..., 0] = 0.282095  # Y_0^0
	
	if degree >= 1:
		# Degree 1
		sh[..., 1] = 0.488603 * y  # Y_1^{-1}
		sh[..., 2] = 0.488603 * z  # Y_1^0
		sh[..., 3] = 0.488603 * x  # Y_1^1
	
	if degree >= 2:
		# Degree 2
		sh[..., 4] = 1.092548 * x * y  # Y_2^{-2}
		sh[..., 5] = 1.092548 * y * z  # Y_2^{-1}
		sh[..., 6] = 0.315392 * (3.0 * z * z - 1.0)  # Y_2^0
		sh[..., 7] = 1.092548 * x * z  # Y_2^1
		sh[..., 8] = 0.546274 * (x * x - y * y)  # Y_2^2
	
	# Higher degrees can be added if needed
	# For now, we support up to degree 2 (9 coefficients)
	
	return sh


def eval_sh_color(sh_coeffs: torch.Tensor, view_dirs: torch.Tensor) -> torch.Tensor:
	"""
	Evaluate spherical harmonics to get RGB colors for given view directions.
	Works with batched inputs.
	
	Args:
		sh_coeffs: SH coefficients tensor
		           - Shape (K, 3) for single Gaussian
		           - Shape (N, K, 3) for batched Gaussians
		           where K = (degree+1)^2
		view_dirs: Normalized view directions
		          - Shape (3,) for single direction
		          - Shape (N, 3) for batched directions
		          Must be normalized!
	
	Returns:
		RGB colors in [0, 1]
		- Shape (3,) for single input
		- Shape (N, 3) for batched input
	"""
	# Determine if batched
	is_batched = len(sh_coeffs.shape) == 3
	
	if is_batched:
		# sh_coeffs: (N, K, 3)
		N, K, _ = sh_coeffs.shape
	else:
		# sh_coeffs: (K, 3)
		K = sh_coeffs.shape[0]
	
	# Determine degree from number of coeffs
	degree = int(math.sqrt(K)) - 1
	
	if degree == 0:
		# Degree 0: Constant color (DC term only)
		constant_factor = 0.282095
		if is_batched:
			color = constant_factor * sh_coeffs[:, 0]
		else:
			color = constant_factor * sh_coeffs[0]
	else:
		# Extract view direction components
		if is_batched:
			x, y, z = view_dirs[:, 0], view_dirs[:, 1], view_dirs[:, 2]
		else:
			x, y, z = view_dirs[0], view_dirs[1], view_dirs[2]
		
		# Evaluate SH basis
		sh_basis = evaluate_sh_basis(x, y, z, degree, device=sh_coeffs.device)
		
		# Apply basis to coefficients
		if is_batched:
			# sh_coeffs: (N, K, 3), sh_basis: (N, K)
			color = torch.sum(sh_coeffs * sh_basis.unsqueeze(-1), dim=1)
		else:
			# sh_coeffs: (K, 3), sh_basis: (K,)
			color = torch.sum(sh_coeffs * sh_basis.unsqueeze(-1), dim=0)
	
	# Apply Sigmoid to map logit -> [0, 1]
	return torch.sigmoid(color)


@dataclass
class GaussianPrimitive:
	"""
	A single 3D Gaussian primitive for splatting.
	
	Each Gaussian is defined by:
	- Position: 3D location in world space
	- Spherical Harmonics: View-dependent color coefficients (per RGB channel)
	- Scale: 3D scale parameters (stored as log scale for optimization)
	- Rotation: Quaternion representation (4 values: w, x, y, z)
	- Opacity: Alpha value (stored as logit for optimization)
	
	Spherical Harmonics:
	- Degree 0: 1 coeff/channel = 3 total (constant color, no view-dependence)
	- Degree 1: 4 coeffs/channel = 12 total (linear view-dependence)
	- Degree 2: 9 coeffs/channel = 27 total (quadratic view-dependence)
	- Default: Degree 0 (RGB only)
	"""
	
	# Position in 3D space (x, y, z)
	position: torch.Tensor  # shape: (3,)
	
	# Spherical harmonics coefficients for RGB
	# Shape: (num_coeffs, 3) where num_coeffs = (degree+1)^2
	# For degree 0: shape (1, 3) - just RGB
	# For degree 1: shape (4, 3) - view-dependent
	# For degree 2: shape (9, 3) - more view-dependent
	sh_coeffs: torch.Tensor  # shape: (num_coeffs, 3)
	
	# Scale parameters (stored as log scale for optimization)
	# After exp: actual scale in 3D (sx, sy, sz)
	scale: torch.Tensor  # shape: (3,)
	
	# Rotation as quaternion (w, x, y, z)
	# Represents rotation of the Gaussian ellipsoid
	rotation: torch.Tensor  # shape: (4,)
	
	# Opacity (stored as logit for optimization)
	# After sigmoid: actual opacity in [0, 1]
	opacity: torch.Tensor  # shape: () - scalar tensor
	
	def __post_init__(self) -> None:
		"""Validate shapes and types."""
		assert self.position.shape == (3,), f"Position must be shape (3,), got {self.position.shape}"
		assert len(self.sh_coeffs.shape) == 2, f"SH coeffs must be 2D, got shape {self.sh_coeffs.shape}"
		assert self.sh_coeffs.shape[1] == 3, f"SH coeffs must have 3 channels (RGB), got {self.sh_coeffs.shape[1]}"
		assert self.scale.shape == (3,), f"Scale must be shape (3,), got {self.scale.shape}"
		assert self.rotation.shape == (4,), f"Rotation must be shape (4,), got {self.rotation.shape}"
		
		# Normalize quaternion
		norm = torch.norm(self.rotation)
		if norm > 0:
			self.rotation = self.rotation / norm
	
	@property
	def sh_degree(self) -> int:
		"""Get the spherical harmonics degree."""
		return get_sh_degree(self.sh_coeffs.shape[0])
	
	def get_color(self, view_dir: torch.Tensor | None = None) -> torch.Tensor:
		"""
		Evaluate spherical harmonics to get RGB color for a given view direction.
		
		Args:
			view_dir: Normalized view direction (3,) in world space. 
			         If None, returns DC term (degree 0) color only.
		
		Returns:
			RGB color (3,) in [0, 1]
		"""
		if view_dir is None:
			# Return DC term only
			view_dir = torch.zeros(3, device=self.sh_coeffs.device)
			view_dir[2] = 1.0  # arbitrary direction for degree 0
		
		# Normalize view direction
		view_dir = view_dir / (torch.norm(view_dir) + 1e-8)
		
		# Use the shared evaluation function
		return eval_sh_color(self.sh_coeffs, view_dir)
	
	@classmethod
	def from_point3d(
		cls,
		position: Union[torch.Tensor, np.ndarray, list, tuple],
		color: Union[torch.Tensor, np.ndarray, list, tuple],
		initial_scale: float = 0.01,
		sh_degree: int = 0,
		device: str | torch.device | None = None,
	) -> GaussianPrimitive:
		"""
		Create a Gaussian primitive from a COLMAP 3D point.
		
		Args:
			position: 3D position (x, y, z)
			color: RGB color (r, g, b) in [0, 255] or [0, 1]
			initial_scale: Initial scale for the Gaussian (will be converted to log scale)
			sh_degree: Spherical harmonics degree (0=RGB only, 1=view-dependent, 2=more view-dependent)
			device: Device to create tensors on (e.g., 'cuda', 'cpu')
		
		Returns:
			GaussianPrimitive with initialized parameters
		"""
		# Convert to torch tensor if needed
		if isinstance(position, (list, tuple, np.ndarray)):
			position = torch.tensor(position, dtype=torch.float32, device=device)
		else:
			position = position.to(device) if device is not None else position
		
		if isinstance(color, (list, tuple, np.ndarray)):
			color = torch.tensor(color, dtype=torch.float32, device=device)
		else:
			color = color.to(device) if device is not None else color
		
		# Normalize color to [0, 1] if it's in [0, 255]
		if color.max() > 1.0:
			color = color / 255.0
		
		# Initialize spherical harmonics coefficients
		# For degree 0: just store RGB in DC term
		# For higher degrees: initialize DC term with color, others to zero
		num_coeffs = get_num_sh_coeffs(sh_degree)
		sh_coeffs = torch.zeros((num_coeffs, 3), dtype=torch.float32, device=position.device)
		
		# Initialize DC term (Y_0^0) with the RGB color
		# Convert from [0, 1] to logit space for optimization
		sh_coeffs[0] = torch.logit(color.clamp(1e-8, 1.0 - 1e-8))
		
		# Initialize scale as log scale (small initial size)
		scale = torch.log(torch.tensor([initial_scale] * 3, dtype=torch.float32, device=position.device))
		
		# Initialize rotation as identity quaternion (w=1, x=0, y=0, z=0)
		rotation = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=position.device)
		
		# Initialize opacity as logit (sigmoid(0) = 0.5)
		opacity = torch.tensor(0.0, dtype=torch.float32, device=position.device)
		
		return cls(
			position=position,
			sh_coeffs=sh_coeffs,
			scale=scale,
			rotation=rotation,
			opacity=opacity,
		)
	
	def get_actual_scale(self) -> torch.Tensor:
		"""Get the actual scale (exp of log scale)."""
		return torch.exp(self.scale)
	
	def get_actual_opacity(self) -> torch.Tensor:
		"""Get the actual opacity (sigmoid of logit)."""
		return torch.sigmoid(self.opacity)
	
	def to_dict(self) -> dict:
		"""Convert to dictionary for serialization."""
		return {
			"position": self.position.detach().cpu().tolist(),
			"sh_coeffs": self.sh_coeffs.detach().cpu().tolist(),
			"sh_degree": self.sh_degree,
			"scale": self.scale.detach().cpu().tolist(),
			"rotation": self.rotation.detach().cpu().tolist(),
			"opacity": float(self.opacity.detach().cpu().item()),
		}
	
	@classmethod
	def from_dict(cls, data: dict, device: str | torch.device | None = None) -> GaussianPrimitive:
		"""Create from dictionary."""
		return cls(
			position=torch.tensor(data["position"], dtype=torch.float32, device=device),
			sh_coeffs=torch.tensor(data["sh_coeffs"], dtype=torch.float32, device=device),
			scale=torch.tensor(data["scale"], dtype=torch.float32, device=device),
			rotation=torch.tensor(data["rotation"], dtype=torch.float32, device=device),
			opacity=torch.tensor(data["opacity"], dtype=torch.float32, device=device),
		)


@dataclass
class Camera:
	"""
	A camera for rendering with intrinsics and extrinsics.
	"""
	# Extrinsics: rotation matrix (3, 3) and translation vector (3,)
	rotation: torch.Tensor  # shape: (3, 3) - rotation matrix
	translation: torch.Tensor  # shape: (3,) - translation vector
	
	# Intrinsics: focal lengths and principal point
	fx: float  # focal length in x direction
	fy: float  # focal length in y direction
	cx: float  # principal point x coordinate
	cy: float  # principal point y coordinate
	
	# Image dimensions
	width: int
	height: int

	def __post_init__(self) -> None:
		"""Validate shapes and types."""
		assert self.rotation.shape == (3, 3), f"Rotation must be shape (3, 3), got {self.rotation.shape}"
		assert self.translation.shape == (3,), f"Translation must be shape (3,), got {self.translation.shape}"
		assert self.fx > 0, f"fx must be positive, got {self.fx}"
		assert self.fy > 0, f"fy must be positive, got {self.fy}"
		assert self.width > 0, f"width must be positive, got {self.width}"
		assert self.height > 0, f"height must be positive, got {self.height}"

	@classmethod
	def from_colmap(cls, colmap_camera, colmap_image, device: str | torch.device | None = None) -> Camera:
		"""
		Create a rendering Camera from COLMAP Camera and Image.
		
		Args:
			colmap_camera: COLMAP Camera namedtuple (from read_cameras_binary)
			colmap_image: COLMAP Image namedtuple (from read_images_binary)
			device: Device to create tensors on
		
		Returns:
			Camera object with intrinsics and extrinsics
		"""
		# Convert quaternion to rotation matrix
		R = colmap_image.qvec2rotmat()
		
		# Extract intrinsics based on camera model
		if colmap_camera.model == "SIMPLE_RADIAL":
			f, cx, cy, k = colmap_camera.params
			fx = fy = f
		elif colmap_camera.model == "PINHOLE":
			fx, fy, cx, cy = colmap_camera.params
		elif colmap_camera.model == "SIMPLE_PINHOLE":
			f, cx, cy = colmap_camera.params
			fx = fy = f
		else:
			raise ValueError(f"Unsupported camera model: {colmap_camera.model}")
		
		# Convert to torch tensors
		if device is not None:
			R = torch.tensor(R, dtype=torch.float32, device=device)
			t = torch.tensor(colmap_image.tvec, dtype=torch.float32, device=device)
		else:
			R = torch.tensor(R, dtype=torch.float32)
			t = torch.tensor(colmap_image.tvec, dtype=torch.float32)
		
		return cls(
			rotation=R,
			translation=t,
			fx=float(fx),
			fy=float(fy),
			cx=float(cx),
			cy=float(cy),
			width=int(colmap_camera.width),
			height=int(colmap_camera.height),
		)


@dataclass
class Image:
	"""
	An image for rendering.
	"""
	id: int
	filename: str
	camera_id: int
	extrinsics: torch.Tensor | None = None  # shape: (4, 4) - optional, can be None if R and t are stored in Camera

