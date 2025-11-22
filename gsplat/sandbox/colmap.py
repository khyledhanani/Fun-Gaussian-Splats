from __future__ import annotations

from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from gsplat.methods.original.original import OriginalGaussianSplat
from gsplat.datasets import ColmapDataset


def main() -> None:
	colmap_path = Path("gsplat/datasets/nerf_llff_data/fern/sparse/0")
	images_path = Path("gsplat/datasets/nerf_llff_data/fern/images")

	print("Loading COLMAP dataset...")
	dataset = ColmapDataset(colmap_path=colmap_path, images_path=images_path)

	print("Initializing OriginalGaussianSplat...")
	method = OriginalGaussianSplat(config={"device": "cuda" if torch.cuda.is_available() else "cpu"})
	method.create_primitives(dataset.get_points3d())
	method.load_cameras_from_dataset(dataset)

	print(f"Total primitives: {len(method.primitives)}")
	sample_idx = 0
	print(f"Rendering sample {sample_idx}...")
	output = method.render_dataset_sample(dataset, idx=sample_idx)

	rendered_image = output["image"]

	# Load target image for loss computation
	target_sample = dataset[sample_idx]
	target_image = Image.open(target_sample["image_path"]).convert("RGB")
	target_tensor = torch.from_numpy(np.array(target_image, dtype=np.float32) / 255.0).to(rendered_image.device)

	loss = F.mse_loss(rendered_image, target_tensor)
	print(f"MSE loss against ground truth: {loss.item():.6f}")

	image = rendered_image.detach().cpu().clamp(0.0, 1.0)

	save_dir = Path("sandbox_outputs")
	save_dir.mkdir(parents=True, exist_ok=True)
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	save_path = save_dir / f"fern_render_{timestamp}.png"

	# Convert to uint8 and save using PIL
	image_uint8 = (image.numpy() * 255).astype("uint8")
	Image.fromarray(image_uint8).save(save_path)
	print(f"Saved render to {save_path}")


if __name__ == "__main__":
	main()