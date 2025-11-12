from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from gsplat.methods.base import GaussianSplatMethod
from gsplat.registry import METHOD_REGISTRY


@METHOD_REGISTRY.register("original")
class OriginalGaussianSplat(GaussianSplatMethod):
	def __init__(self, config: Dict[str, Any]) -> None:
		super().__init__(config)
		self.state: Dict[str, Any] = {}

	def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
		# TODO: implement original Gaussian Splatting optimization here
		return {"loss": 0.0}

	def render(self, camera: Dict[str, Any]) -> Dict[str, Any]:
		# TODO: implement forward rendering using current Gaussian set
		return {"image": None}

	def save(self, path: Path) -> None:
		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_bytes(b"placeholder")

	def load(self, path: Path, strict: bool = True) -> None:
		_ = path.read_bytes()


