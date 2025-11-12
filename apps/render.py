from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

from gsplat.registry import METHOD_REGISTRY
from gsplat.utils import create_logger


def load_config(path: str | Path) -> Dict[str, Any]:
	with open(path, "r") as f:
		return yaml.safe_load(f)


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--checkpoint", type=str, required=True)
	parser.add_argument("--config", type=str, default="configs/original.yaml")
	parser.add_argument("--output", type=str, required=True)
	args = parser.parse_args()

	config = load_config(args.config)
	logger = create_logger("apps.render")

	method_type = METHOD_REGISTRY.get(config.get("method", "original"))
	method = method_type(config=config)

	ckpt_path = Path(args.checkpoint)
	method.load(ckpt_path, strict=False)

	# Placeholder camera
	camera: Dict[str, Any] = {"pose": None, "intrinsics": None}
	result = method.render(camera)

	out_dir = Path(args.output)
	out_dir.mkdir(parents=True, exist_ok=True)
	(out_dir / "image.txt").write_text("placeholder_image")

	logger.info(f"Rendered image to {out_dir}")


if __name__ == "__main__":
	main()


