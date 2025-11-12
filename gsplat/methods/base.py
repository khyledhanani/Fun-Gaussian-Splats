from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional


class GaussianSplatMethod(ABC):
	"""
	Base interface all methods must implement.
	Encapsulates training, rendering, and checkpoint IO.
	"""

	def __init__(self, config: Dict[str, Any]) -> None:
		self.config = config

	@abstractmethod
	def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Run one optimization step and return logs (e.g., loss).
		"""

	@abstractmethod
	def render(self, camera: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Render outputs from a camera specification (pose, intrinsics).
		"""

	@abstractmethod
	def save(self, path: Path) -> None:
		"""
		Save method state to path.
		"""

	@abstractmethod
	def load(self, path: Path, strict: bool = True) -> None:
		"""
		Load method state from path.
		"""


