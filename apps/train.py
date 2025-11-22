from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from PIL import Image as PILImage

from gsplat.datasets import ColmapDataset
from gsplat.methods.base import GaussianSplatMethod
from gsplat.registry import METHOD_REGISTRY
from gsplat.utils import create_logger, set_global_seed


def load_config(path: str | Path) -> Dict[str, Any]:
	with open(path, "r") as f:
		return yaml.safe_load(f)


def build_method(config: Dict[str, Any]) -> GaussianSplatMethod:
	name = config.get("method", "original")
	method_type = METHOD_REGISTRY.get(name)
	return method_type(config=config)


def load_image(image_path: Path, device: str = "cpu") -> torch.Tensor:
	"""Load image and convert to tensor."""
	img = PILImage.open(image_path).convert("RGB")
	img_array = torch.from_numpy(torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))).float()
	img_array = img_array.view(img.size[1], img.size[0], 3) / 255.0
	return img_array.to(device)


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
	args = parser.parse_args()

	config = load_config(args.config)
	logger = create_logger("apps.train")

	seed = int(config.get("seed", 42))
	set_global_seed(seed)

	# Load dataset
	data_config = config.get("data", {})
	colmap_path = Path(data_config.get("colmap_path", "./data/colmap/sparse/0"))
	images_path = Path(data_config.get("images_path", "./data/images"))
	
	logger.info(f"Loading dataset from {colmap_path}")
	dataset = ColmapDataset(
		colmap_path=colmap_path,
		images_path=images_path,
		split=data_config.get("split", "train"),
		device=config.get("device", "cpu"),
	)
	logger.info(f"Loaded {len(dataset)} images")

	# Build method and initialize with dataset
	method = build_method(config)
	logger.info("Loading cameras and primitives...")
	method.load_cameras_from_dataset(dataset)
	method.create_primitives(dataset.get_points3d())
	method.setup_optimizer()
	logger.info("Initialization complete")

	# Setup output directory
	runs_dir = Path(config.get("runs_dir", "runs")) / config.get("run_name", "debug")
	runs_dir.mkdir(parents=True, exist_ok=True)
	(runs_dir / "config.json").write_text(json.dumps(config, indent=2))

	# Training loop
	max_steps = int(config.get("trainer", {}).get("max_steps", 10))
	log_every = int(config.get("trainer", {}).get("log_every", 1))
	
	logger.info(f"Starting training for {max_steps} steps")
	
	for step in range(max_steps):
		# Randomly sample a camera/image
		idx = random.randint(0, len(dataset) - 1)
		sample = dataset[idx]
		
		# Load target image
		target_image = load_image(sample["image_path"], device=method.device)
		
		# Create batch
		batch = {
			"camera": sample["camera"],
			"target_image": target_image,
			"background_color": config.get("background_color", (0.0, 0.0, 0.0)),
		}
		
		# Training step
		logs = method.train_step(batch)
		
		if step % log_every == 0:
			logger.info(f"step={step} loss={logs['loss']:.6f} num_gaussians={method.means.shape[0]}")

	# Save checkpoint
	ckpt_path = runs_dir / "checkpoint.pt"
	method.save(ckpt_path)
	logger.info(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
	main()


