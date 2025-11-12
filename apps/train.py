from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml

from gsplat.datasets import NerfDataset
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


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
	args = parser.parse_args()

	config = load_config(args.config)
	logger = create_logger("apps.train")

	seed = int(config.get("seed", 42))
	set_global_seed(seed)

	train_root = config.get("data", {}).get("train_root", "./data/train")
	dataset = NerfDataset(train_root, split="train")

	method = build_method(config)

	runs_dir = Path(config.get("runs_dir", "runs")) / config.get("run_name", "debug")
	runs_dir.mkdir(parents=True, exist_ok=True)
	(runs_dir / "config.json").write_text(json.dumps(config, indent=2))

	max_steps = int(config.get("trainer", {}).get("max_steps", 10))
	for step, batch in zip(range(max_steps), dataset):
		logs = method.train_step(batch)
		if step % int(config.get("trainer", {}).get("log_every", 1)) == 0:
			logger.info(f"step={step} logs={logs}")

	ckpt_path = runs_dir / "checkpoint.pt"
	method.save(ckpt_path)
	logger.info(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
	main()


