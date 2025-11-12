from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from gsplat.methods.base import GaussianSplatMethod
from gsplat.registry import METHOD_REGISTRY


@METHOD_REGISTRY.register("vq")
class VectorQuantizedGaussianSplat(GaussianSplatMethod):
	def __init__(self, config: Dict[str, Any]) -> None:
		super().__init__(config)
		self.state: Dict[str, Any] = {"codebook": None}

	def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
		# TODO: implement VQ bottleneck + commitment loss + codebook updates
		return {"loss": 0.0, "vq_loss": 0.0}

	def render(self, camera: Dict[str, Any]) -> Dict[str, Any]:
		# TODO: render conditioned on quantized vectors
		return {"image": None}

	def save(self, path: Path) -> None:
		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_bytes(b"placeholder_vq")

	def load(self, path: Path, strict: bool = True) -> None:
		_ = path.read_bytes()


