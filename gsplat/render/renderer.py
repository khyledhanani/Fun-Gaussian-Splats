from __future__ import annotations

from typing import Any, Dict


class Renderer:
	"""
	Stub renderer. Replace with CPU reference and later CUDA kernels (in csrc/).
	"""

	def __init__(self, config: Dict[str, Any]) -> None:
		self.config = config

	def forward(self, scene: Dict[str, Any], camera: Dict[str, Any]) -> Dict[str, Any]:
		# TODO: implement rasterization of Gaussians
		return {"image": None}


