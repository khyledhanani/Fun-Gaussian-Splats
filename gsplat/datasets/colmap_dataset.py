from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Iterator, List
from PIL import Image as PILImage

from gsplat.datasets.colmap import (
	read_cameras_binary,
	read_images_binary,
	read_points3D_binary,
	Camera as ColmapCamera,
	Image as ColmapImage,
	Point3D,
)
from gsplat.utils.dataclasses import Camera, Image

"""
dataset loader written by legendary claude sonnet
"""


class ColmapDataset:
	"""
	Dataset loader for COLMAP reconstruction outputs.
	
	Reads COLMAP's sparse reconstruction format:
	- cameras.bin or cameras.txt (camera intrinsics)
	- images.bin or images.txt (camera poses + image paths)
	- points3D.bin or points3D.txt (sparse 3D point cloud for initialization)
	
	This is the standard input format for Gaussian Splatting, as it provides
	both camera poses and initial 3D points for Gaussian initialization.
	"""

	def __init__(
		self, 
		colmap_path: str | Path, 
		images_path: str | Path, 
		split: str = "train",
		device: str | None = None,
	) -> None:
		"""
		Args:
			colmap_path: Path to COLMAP sparse reconstruction directory (contains cameras/images/points3D)
			images_path: Path to directory containing the actual image files
			split: Dataset split ("train", "val", "test")
			device: Device to create tensors on (e.g., 'cuda', 'cpu')
		"""
		self.colmap_path = Path(colmap_path)
		self.images_path = Path(images_path)
		self.split = split
		self.device = device
		
		# Load COLMAP data from binary files
		self.colmap_cameras: Dict[int, ColmapCamera] = {}
		self.colmap_images: Dict[int, ColmapImage] = {}
		self.points3d: Dict[int, Point3D] = {}
		
		# Converted rendering cameras and images
		self.cameras: List[Camera] = []
		self.images: List[Image] = []
		self.samples: List[Dict[str, Any]] = []
		
		# Load the data
		self._load_colmap_data()

	def _load_colmap_data(self) -> None:
		"""Load cameras, images, and points from COLMAP binary files."""
		# Determine if we should use .bin or .txt files
		cameras_bin = self.colmap_path / "cameras.bin"
		images_bin = self.colmap_path / "images.bin"
		points3d_bin = self.colmap_path / "points3D.bin"
		
		if cameras_bin.exists() and images_bin.exists() and points3d_bin.exists():
			# Load from binary files
			self.colmap_cameras = read_cameras_binary(cameras_bin)
			self.colmap_images = read_images_binary(images_bin)
			self.points3d = read_points3D_binary(points3d_bin)
		else:
			raise FileNotFoundError(
				f"COLMAP binary files not found in {self.colmap_path}. "
				"Expected: cameras.bin, images.bin, points3D.bin"
			)
		
		# Convert COLMAP cameras/images to rendering format
		self._convert_cameras_and_images()
		
		# Create samples for iteration
		self._create_samples()

	def _convert_cameras_and_images(self) -> None:
		"""Convert COLMAP Camera/Image to rendering Camera/Image format."""
		self.cameras = []
		self.images = []
		
		# Sort images by ID for consistent ordering
		sorted_image_ids = sorted(self.colmap_images.keys())

		# Check if we need to map filenames (e.g. if using images_4 with different names)
		colmap_filenames = [self.colmap_images[i].name for i in sorted_image_ids]
		first_img_path = self.images_path / colmap_filenames[0]
		
		filename_map = {}
		if not first_img_path.exists():
			print(f"Notice: Could not find {colmap_filenames[0]} in {self.images_path}")
			print("Attempting to match images by sorted order...")
			
			# Get all image files in directory
			image_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".PNG"}
			dir_files = sorted([
				f.name for f in self.images_path.iterdir() 
				if f.suffix in image_extensions and not f.name.startswith(".")
			])
			
			# Filter COLMAP images to only those that seem to be in this set
			# (Simple assumption: if counts match, we map 1:1 on sorted names)
			sorted_colmap_names = sorted(colmap_filenames)
			
			if len(dir_files) == len(colmap_filenames):
				print(f"Success: Matched {len(dir_files)} images by sorted order.")
				for c_name, d_name in zip(sorted_colmap_names, dir_files):
					filename_map[c_name] = d_name
			else:
				print(f"Warning: Image count mismatch. COLMAP: {len(colmap_filenames)}, Dir: {len(dir_files)}")
				print("Will try to load with original filenames (might fail).")
		
		for img_id in sorted_image_ids:
			colmap_image = self.colmap_images[img_id]
			
			# Resolve filename using map if available
			filename = filename_map.get(colmap_image.name, colmap_image.name)

			colmap_camera = self.colmap_cameras[colmap_image.camera_id]
			
			# Convert to rendering Camera
			camera = Camera.from_colmap(
				colmap_camera=colmap_camera,
				colmap_image=colmap_image,
				device=self.device,
			)

			# Handle downsampled images (e.g. images_4)
			# If the actual image file is smaller than COLMAP metadata, scale intrinsics
			image_path = self.images_path / filename
			if image_path.exists():
				with PILImage.open(image_path) as img:
					actual_width, actual_height = img.size
				
				if actual_width != camera.width or actual_height != camera.height:
					scale_x = actual_width / camera.width
					scale_y = actual_height / camera.height
					
					camera.fx *= scale_x
					camera.fy *= scale_y
					camera.cx *= scale_x
					camera.cy *= scale_y
					camera.width = actual_width
					camera.height = actual_height

			self.cameras.append(camera)
			
			# Create Image dataclass
			image = Image(
				id=colmap_image.id,
				filename=filename,
				camera_id=colmap_image.camera_id,
				extrinsics=None,  # We store R and t separately in Camera
			)
			self.images.append(image)

	def _create_samples(self) -> None:
		"""Create sample dictionaries for iteration."""
		self.samples = []
		for camera, image in zip(self.cameras, self.images):
			sample = {
				"camera": camera,
				"image": image,
				"image_path": self.images_path / image.filename,
			}
			self.samples.append(sample)

	def __len__(self) -> int:
		return len(self.samples)

	def __iter__(self) -> Iterator[Dict[str, Any]]:
		for sample in self.samples:
			yield sample

	def __getitem__(self, idx: int) -> Dict[str, Any]:
		"""Get a sample by index."""
		return self.samples[idx]

	def get_points3d(self) -> Dict[int, Point3D]:
		"""
		Return the sparse 3D point cloud for Gaussian initialization.
		Returns dict mapping point_id to Point3D objects from COLMAP.
		"""
		return self.points3d
	
	def get_cameras(self) -> List[Camera]:
		"""Return list of rendering Camera objects."""
		return self.cameras
	
	def get_images(self) -> List[Image]:
		"""Return list of Image objects."""
		return self.images

